# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Small, explicitly owned conversion pool with bounded worker lifetimes.

Fresh interpreters receive their environment before importing ai4rag or Docling.
Only configuration, paths and small results cross the private control sockets;
S3 clients and converters never cross a process boundary.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Iterator

from ._extraction_protocol import _INCOMPLETE, _read_message, _send_message

_SHUTDOWN_TIMEOUT = 5.0
_WORKER_MEMORY_EXIT_CODE = 75
_logger = logging.getLogger("text-extraction")
_WORKER_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DISABLE": "1",
}


def _worker_command(descriptor: int) -> list[str]:
    return [sys.executable, "-u", "-m", "ai4rag.utils.data._extraction_worker", str(descriptor)]


@dataclass
class _Worker:
    process: subprocess.Popen
    connection: socket.socket
    task: tuple[Path, str] | None = None
    deadline: float = 0.0
    response: bytearray = field(default_factory=bytearray)


class _WorkerMemoryError(RuntimeError):
    """A worker reported exhausted memory or was killed by SIGKILL."""


class ConversionPool:
    """Own conversion processes, including termination of their subprocesses.

    Used only on the supported POSIX platforms. Each worker starts a new
    session so a timed-out conversion's FFmpeg subprocess is terminated too.
    """

    def __init__(
        self,
        workers: int,
        config: dict[str, Any],
        artifacts_path: Path | None,
        startup_timeout: float,
        conversion_timeout: float,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("Text extraction worker supervision requires a POSIX platform.")
        if workers < 1:
            raise ValueError("Text extraction requires at least one conversion worker.")
        self.workers: list[_Worker] = []
        self.count = workers
        self.config = config
        self.environment = dict(os.environ, **_WORKER_ENV)
        if artifacts_path is not None:
            self.environment["DOCLING_ARTIFACTS_PATH"] = str(artifacts_path.resolve())
            self.environment.setdefault("HF_HUB_OFFLINE", "1")
        self.startup_timeout = startup_timeout
        self.conversion_timeout = conversion_timeout
        self._memory_recovery_used = False
        self._closed = True

    def __enter__(self) -> "ConversionPool":
        if not self._closed or self.workers:
            raise RuntimeError("The text extraction conversion pool can only be entered once.")
        self._start()
        return self

    def _start(self) -> None:
        self._closed = False
        try:
            for _ in range(self.count):
                parent, child = socket.socketpair()
                try:
                    # The pool owns this process across conversions and recovery.
                    process = subprocess.Popen(  # pylint: disable=consider-using-with
                        _worker_command(child.fileno()),
                        pass_fds=(child.fileno(),),
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                        env=self.environment,
                    )
                    worker = _Worker(process, parent, deadline=time.monotonic() + self.startup_timeout)
                    self.workers.append(worker)
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._send_to_worker(worker, self.config, "initialization")
            pending = list(self.workers)
            while pending:
                for worker, message in self._responses(pending, "initialization"):
                    if message != ["ready"]:
                        raise RuntimeError(f"Text extraction worker initialization failed: {message!r}")
                    pending.remove(worker)
        except BaseException as exc:
            self._close(graceful=False, primary_error=exc)
            raise

    def convert(self, documents: list[tuple[Path, str]], output_dir: Path) -> Iterator[tuple[Path, bool, str | None]]:
        """Convert a batch, allowing one memory-related pool restart per run.

        Completed results are removed before yielding so restarting the pool
        retries only unfinished documents, including interrupted conversions.
        """
        if self._closed or not self.workers:
            raise RuntimeError("Text extraction conversion requires a running worker pool.")
        remaining = list(documents)
        while True:
            try:
                yield from self._convert(remaining, output_dir)
                return
            except _WorkerMemoryError as exc:
                if self._memory_recovery_used:
                    raise RuntimeError(f"{exc} Text extraction exhausted its single memory-recovery attempt.") from exc
                self._memory_recovery_used = True
                _logger.warning(
                    "%s Restarting the conversion pool once and retrying %d unfinished document(s).",
                    exc,
                    len(remaining),
                )
                # Reap the entire old pool before loading replacement model copies.
                if not self._close(graceful=False, primary_error=exc):
                    raise
                self.workers.clear()
                self._start()

    def _convert(self, remaining: list[tuple[Path, str]], output_dir: Path) -> Iterator[tuple[Path, bool, str | None]]:
        """Consume one attempt, removing acknowledged tasks from ``remaining``."""
        pending = iter(list(remaining))
        active: list[_Worker] = []
        for worker in self.workers:
            if self._submit(worker, pending, output_dir):
                active.append(worker)
        while active:
            for worker, message in self._responses(active, "conversion"):
                if not isinstance(message, list) or len(message) != 2 or not isinstance(message[0], bool):
                    raise RuntimeError(f"Invalid text extraction worker result: {message!r}")
                path, _key = worker.task
                remaining.remove(worker.task)
                worker.task = None
                yield path, message[0], message[1]
                if not self._submit(worker, pending, output_dir):
                    active.remove(worker)

    def _submit(self, worker: _Worker, pending: Iterator[tuple[Path, str]], output_dir: Path) -> bool:
        task = next(pending, None)
        if task is None:
            return False
        worker.task = task
        worker.deadline = time.monotonic() + self.conversion_timeout
        self._send_to_worker(worker, (str(task[0]), str(output_dir), task[1]), "conversion")
        return True

    def _send_to_worker(self, worker: _Worker, message: Any, stage: str) -> None:
        try:
            _send_message(worker.connection, message, timeout=max(0.0, worker.deadline - time.monotonic()))
        except (TimeoutError, BlockingIOError) as exc:
            raise self._timeout_error(worker, stage) from exc
        except OSError as exc:
            self._check_worker(worker, wait_for_exit=True)
            raise RuntimeError(f"Text extraction worker pid={worker.process.pid} stopped before {stage}.") from exc

    def _responses(self, workers: list[_Worker], stage: str) -> Iterator[tuple[_Worker, Any]]:
        for worker in self.workers:
            self._check_worker(worker)
        timeout = min(0.05, max(0.0, min(worker.deadline for worker in workers) - time.monotonic()))
        connections = set(wait([worker.connection for worker in workers], timeout=timeout))
        now = time.monotonic()
        for worker in list(workers):
            # A completed response may have waited while the parent published
            # another document. Parent-side I/O is not conversion runtime.
            if worker.connection in connections:
                try:
                    message = _read_message(worker.connection, worker.response, nonblocking=True)
                except (EOFError, OSError) as exc:
                    self._check_worker(worker, wait_for_exit=True)
                    raise RuntimeError(
                        f"Text extraction worker pid={worker.process.pid} closed its control socket."
                    ) from exc
                except ValueError as exc:
                    raise RuntimeError(f"Invalid text extraction worker response during {stage}.") from exc
                if message is not _INCOMPLETE:
                    yield worker, message
                    continue
            if now >= worker.deadline:
                raise self._timeout_error(worker, stage)

    @staticmethod
    def _timeout_error(worker: _Worker, stage: str) -> RuntimeError:
        key = worker.task[1] if worker.task else "no active document"
        return RuntimeError(f"Text extraction worker {stage} timed out ({key}).")

    @staticmethod
    def _check_worker(worker: _Worker, wait_for_exit: bool = False) -> None:
        exitcode = worker.process.poll()
        if exitcode is None and wait_for_exit:
            # Pipe closure can precede the process becoming waitable. Give it a
            # bounded interval to expose the exit status before classifying it.
            try:
                exitcode = worker.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                return
        if exitcode is None:
            return
        key = worker.task[1] if worker.task else "no active document"
        message = f"Text extraction worker pid={worker.process.pid} exited with code {exitcode} ({key})."
        if exitcode == _WORKER_MEMORY_EXIT_CODE:
            raise _WorkerMemoryError(f"{message} The worker reported MemoryError.")
        if exitcode == -signal.SIGKILL:
            raise _WorkerMemoryError(f"{message} SIGKILL may indicate an out-of-memory kill.")
        raise RuntimeError(message)

    def __exit__(self, _exc_type: Any, exc: BaseException | None, _tb: Any) -> None:
        self._close(graceful=exc is None, primary_error=exc)

    def _close(self, graceful: bool, primary_error: BaseException | None = None) -> bool:
        """Attempt cleanup once and record its failures on an active exception."""
        if self._closed:
            return True
        self._closed = True
        errors: list[str] = []
        try:
            if graceful:
                deadline = time.monotonic() + _SHUTDOWN_TIMEOUT
                for worker in self.workers:
                    try:
                        _send_message(worker.connection, None, timeout=max(0.0, deadline - time.monotonic()))
                    except OSError:
                        pass
                self._wait_for_exit(errors)
            # Signal groups even when a leader has stopped, since its decoder can still be running.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                self._signal_groups(sig, errors)
                unreaped = self._wait_for_exit(errors)
            if unreaped:
                errors.append(f"Unable to reap text extraction workers: {unreaped}")
        finally:
            for worker in self.workers:
                try:
                    worker.connection.close()
                except OSError as exc:
                    errors.append(f"Closing worker pid={worker.process.pid} control socket failed: {exc}")
        if not errors:
            return True
        details = "Text extraction worker cleanup failed. " + "\n".join(errors)
        if primary_error is None:
            raise RuntimeError(details)
        primary_error.add_note(details)
        return False

    def _signal_groups(self, sig: signal.Signals, errors: list[str]) -> None:
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT
        for worker in self.workers:
            try:
                self._signal_group(worker.process, sig, deadline)
            except ProcessLookupError:
                pass
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"Sending {sig.name} to worker group pid={worker.process.pid} failed: {exc}")

    @staticmethod
    def _signal_group(process: subprocess.Popen, sig: signal.Signals, deadline: float) -> None:
        # macOS can report EPERM for a group containing only a zombie leader.
        process.poll()
        try:
            os.killpg(process.pid, sig)
        except PermissionError:
            # The leader may have exited between poll() and killpg(). Reap and retry.
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
            os.killpg(process.pid, sig)

    def _wait_for_exit(self, errors: list[str]) -> list[int]:
        unreaped = []
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT
        for worker in self.workers:
            try:
                worker.process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                unreaped.append(worker.process.pid)
            except OSError as exc:
                errors.append(f"Waiting for worker pid={worker.process.pid} failed: {exc}")
                unreaped.append(worker.process.pid)
        return unreaped


def _container_cpu_quota() -> float | None:
    """Read Linux cgroup CPU quotas, including the process's ancestor groups."""
    roots = [(Path("/sys/fs/cgroup"), "cpu.max")]
    roots.extend((Path("/sys/fs/cgroup") / name, "cpu.cfs_quota_us") for name in ("cpu", "cpu,cpuacct"))
    try:
        memberships = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        memberships = []
    quotas = []
    for root, filename in roots:
        directories = {root}
        for membership in memberships:
            fields = membership.split(":", 2)
            if len(fields) != 3:
                continue
            _identifier, controllers, relative = fields
            if (filename == "cpu.max" and not controllers) or "cpu" in controllers.split(","):
                directory = root / relative.lstrip("/")
                while directory.is_relative_to(root) and directory != root:
                    directories.add(directory)
                    directory = directory.parent
        for directory in directories:
            try:
                if filename == "cpu.max":
                    quota, period = (directory / filename).read_text(encoding="utf-8").split()
                else:
                    quota = (directory / filename).read_text(encoding="utf-8").strip()
                    period = (directory / "cpu.cfs_period_us").read_text(encoding="utf-8").strip()
                if float(quota) > 0 and float(period) > 0:
                    quotas.append(float(quota) / float(period))
            except (OSError, ValueError):
                continue
    return min(quotas) if quotas else None
