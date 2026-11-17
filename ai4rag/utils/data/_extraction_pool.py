# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Small, explicitly owned conversion pool with bounded worker lifetimes.

Fresh interpreters receive their environment before importing ai4rag or Docling.
Only configuration, paths and small results cross the private control pipes;
S3 clients and converters never cross a process boundary.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Iterator

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
    return [sys.executable, "-m", "ai4rag.utils.data._extraction_worker", str(descriptor)]


@dataclass
class _Worker:
    process: subprocess.Popen
    connection: Connection
    task: tuple[Path, str] | None = None
    deadline: float = 0.0


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
        self._closed = False

    def __enter__(self) -> "ConversionPool":
        self._start()
        return self

    def _start(self) -> None:
        self._closed = False
        try:
            for _ in range(self.count):
                parent, child = Pipe()
                try:
                    # The pool owns this process across conversions and recovery.
                    process = subprocess.Popen(  # pylint: disable=consider-using-with
                        _worker_command(child.fileno()),
                        pass_fds=(child.fileno(),),
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                        env=self.environment,
                    )
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                worker = _Worker(process, parent, deadline=time.monotonic() + self.startup_timeout)
                self.workers.append(worker)
                parent.send(self.config)
            pending = list(self.workers)
            while pending:
                for worker in self._ready(pending, "initialization"):
                    message = self._receive(worker)
                    if message != ("ready",):
                        raise RuntimeError(f"Text extraction worker initialization failed: {message!r}")
                    pending.remove(worker)
        except BaseException:
            self._close(graceful=False)
            raise

    def convert(self, documents: list[tuple[Path, str]], output_dir: Path) -> Iterator[tuple[Path, bool, str | None]]:
        """Convert a batch, allowing one memory-related pool restart per run.

        Completed results are removed before yielding so restarting the pool
        retries only unfinished documents, including interrupted conversions.
        """
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
                self._close(graceful=False)
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
            for worker in self._ready(active, "conversion"):
                message = self._receive(worker)
                if not isinstance(message, tuple) or len(message) != 2 or not isinstance(message[0], bool):
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
        try:
            worker.connection.send((str(task[0]), str(output_dir), task[1]))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._check_worker(worker, wait_for_exit=True)
            raise RuntimeError(f"Text extraction worker pid={worker.process.pid} stopped before conversion.") from exc
        return True

    def _ready(self, workers: list[_Worker], stage: str) -> list[_Worker]:
        for worker in self.workers:
            self._check_worker(worker)
        connections = wait([worker.connection for worker in workers], timeout=0.05)
        now = time.monotonic()
        for worker in workers:
            # A completed response may have waited while the parent published
            # another document. Parent-side I/O is not conversion runtime.
            if worker.connection not in connections and now >= worker.deadline:
                key = worker.task[1] if worker.task else "no active document"
                raise RuntimeError(f"Text extraction worker {stage} timed out ({key}).")
        return [worker for worker in workers if worker.connection in connections]

    def _receive(self, worker: _Worker) -> tuple:
        try:
            return worker.connection.recv()
        except (EOFError, OSError) as exc:
            self._check_worker(worker, wait_for_exit=True)
            raise RuntimeError(f"Text extraction worker pid={worker.process.pid} closed its control pipe.") from exc

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

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        self._close(graceful=exc_type is None)

    def _close(self, graceful: bool) -> None:
        if self._closed:
            return
        try:
            if graceful:
                for worker in self.workers:
                    try:
                        worker.connection.send(None)
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                self._wait_for_exit()
        finally:
            # Signal the entire session even if its worker has already exited:
            # a failed conversion can have left a live FFmpeg child behind.
            try:
                self._signal_groups(signal.SIGTERM)
                self._wait_for_exit()
                self._signal_groups(signal.SIGKILL)
                self._wait_for_exit()
            finally:
                for worker in self.workers:
                    worker.connection.close()
                unreaped = [worker.process.pid for worker in self.workers if worker.process.poll() is None]
                if unreaped:
                    raise RuntimeError(f"Unable to reap text extraction workers: {unreaped}")
                self._closed = True

    def _signal_groups(self, sig: signal.Signals) -> None:
        for worker in self.workers:
            # Reap an already-dead leader before signalling its group. macOS
            # can report EPERM for a group containing only the zombie leader.
            worker.process.poll()
            try:
                os.killpg(worker.process.pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                # The leader can exit between poll() and killpg(). Reap that
                # zombie and retry; do not suppress a real permission failure.
                worker.process.wait(timeout=_SHUTDOWN_TIMEOUT)
                try:
                    os.killpg(worker.process.pid, sig)
                except ProcessLookupError:
                    pass

    def _wait_for_exit(self) -> None:
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT
        for worker in self.workers:
            try:
                worker.process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


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
            _identifier, controllers, relative = membership.split(":", 2)
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
