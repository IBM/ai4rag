# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Exercise real worker processes without loading conversion models."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from ai4rag.utils.data import _extraction_pool as pools

_CHILD = """
import json, os, signal, subprocess, sys, time
from multiprocessing.connection import Connection
from pathlib import Path
with Connection(int(sys.argv[1])) as connection:
    config = connection.recv()
    mode = config.get('mode')
    if config.get('fail_restart') and any(Path(config['markers']).iterdir()):
        raise RuntimeError('replacement initialization failed')
    if mode == 'init_failure':
        raise RuntimeError('initialization failed')
    if mode == 'init_hang':
        time.sleep(60)
    connection.send(('ready',))
    while (task := connection.recv()) is not None:
        if mode in ('memory_exit', 'killed'):
            with open(config['calls'], 'a') as calls:
                calls.write(json.dumps([os.getpid(), task[2]]) + '\\n')
            if config.get('wait_for'):
                if task[2] == 'interrupted' and not (Path(config['markers']) / 'bad').exists():
                    Path(config['wait_for']).touch()
                    time.sleep(0.5)
                if task[2] == 'bad':
                    while not Path(config['wait_for']).exists():
                        time.sleep(0.01)
            if task[2] in config.get('fail_keys', ['bad']):
                marker = Path(config['markers']) / task[2]
                if not marker.exists() or config.get('always_fail'):
                    marker.touch()
                    if mode == 'killed':
                        os.kill(os.getpid(), signal.SIGKILL)
                    os._exit(75)
        if mode == 'death':
            os._exit(23)
        if mode == 'hang':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
        if mode == 'descendant':
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            open(config['pid_file'], 'w').write(str(child.pid))
            time.sleep(60)
        if mode == 'environment':
            assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
            assert os.environ['DOCLING_ARTIFACTS_PATH'] == config['artifacts']
        connection.send((True, str(os.getpid())))
"""


@pytest.fixture
def worker_command(monkeypatch):
    """Use the actual transport and supervisor with a tiny conversion substitute."""
    monkeypatch.setattr(pools, "_worker_command", lambda fd: [sys.executable, "-c", _CHILD, str(fd)])
    monkeypatch.setattr(pools, "_SHUTDOWN_TIMEOUT", 0.5)


def _pool(config=None, **kwargs):
    return pools.ConversionPool(
        workers=kwargs.pop("workers", 1),
        config=config or {},
        artifacts_path=kwargs.pop("artifacts_path", None),
        startup_timeout=kwargs.pop("startup_timeout", 5),
        conversion_timeout=kwargs.pop("conversion_timeout", 5),
        **kwargs,
    )


def _assert_reaped(pool):
    assert pool.workers
    assert all(worker.process.poll() is not None and worker.connection.closed for worker in pool.workers)


def test_reuses_workers_and_bounds_active_work(worker_command, tmp_path):
    """Many tasks are processed by the configured worker processes only."""
    with _pool(workers=2) as pool:
        results = list(pool.convert([(Path(str(i)), str(i)) for i in range(20)], tmp_path))
        first_pids = {pid for _, success, pid in results if success}
        more = list(pool.convert([(Path("another"), "another")], tmp_path))
        assert more[0][2] in first_pids
        assert len(first_pids) <= 2
        assert len(results) == 20
    _assert_reaped(pool)


@pytest.mark.parametrize("mode", ["init_failure", "init_hang", "death", "hang"])
def test_worker_failures_terminate_and_reap(worker_command, tmp_path, mode):
    """Initialization failure, worker loss and stuck work cannot wait forever."""
    pool = _pool({"mode": mode}, startup_timeout=0.5 if mode == "init_hang" else 5, conversion_timeout=0.3)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="worker"):
        with pool:
            list(pool.convert([(Path("input"), "original/key.pdf")], tmp_path))
    assert time.monotonic() - start < 10
    _assert_reaped(pool)
    assert not pool._memory_recovery_used


def test_kills_conversion_descendants(worker_command, tmp_path):
    """A stuck conversion's decoder must not survive worker cleanup."""
    pid_file = tmp_path / "decoder.pid"
    pool = _pool({"mode": "descendant", "pid_file": str(pid_file)}, conversion_timeout=0.5)
    with pytest.raises(RuntimeError, match="conversion timed out"):
        with pool:
            list(pool.convert([(Path("input"), "audio.wav")], tmp_path))
    _assert_reaped(pool)
    decoder_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(decoder_pid) and time.monotonic() < deadline:
        if psutil.Process(decoder_pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.02)
    assert not psutil.pid_exists(decoder_pid) or psutil.Process(decoder_pid).status() == psutil.STATUS_ZOMBIE


def test_child_environment_is_isolated(worker_command, monkeypatch, tmp_path):
    """Limits and explicit artifacts are applied before child imports, not to the caller."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "96")
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", "/other/models")
    with _pool({"mode": "environment", "artifacts": str(tmp_path)}, artifacts_path=tmp_path) as pool:
        assert list(pool.convert([(Path("input"), "input")], tmp_path))[0][1]
    assert os.environ["OPENBLAS_NUM_THREADS"] == "96"
    assert os.environ["DOCLING_ARTIFACTS_PATH"] == "/other/models"


def test_concurrent_calls_have_independent_workers(worker_command, tmp_path):
    """Concurrent callers share no worker or control connection."""

    def run(index):
        with _pool() as pool:
            return list(pool.convert([(Path(str(index)), str(index))], tmp_path))[0][2]

    with ThreadPoolExecutor(2) as executor:
        pids = list(executor.map(run, range(2)))
    assert len(set(pids)) == 2


def test_parent_publication_delay_does_not_timeout_completed_work(worker_command, tmp_path):
    """Completed conversion responses can wait while another artifact is published."""
    with _pool(workers=2, conversion_timeout=0.3) as pool:
        results = pool.convert([(Path(str(i)), str(i)) for i in range(2)], tmp_path)
        assert next(results)[1]
        time.sleep(0.5)
        assert next(results)[1]
        assert list(results) == []


def test_partial_startup_failure_reaps_already_started_workers(worker_command, monkeypatch):
    """Failure to start a later interpreter must clean up the earlier ones."""
    popen = pools.subprocess.Popen
    calls = 0

    def start(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("cannot start process")
        return popen(*args, **kwargs)

    monkeypatch.setattr(pools.subprocess, "Popen", start)
    pool = _pool(workers=2)
    with pytest.raises(OSError, match="cannot start"):
        with pool:
            pytest.fail("Startup should fail")
    _assert_reaped(pool)


def test_caller_exception_reaps_workers(worker_command):
    """Cancellation and parent-side failures follow the same cleanup path."""
    pool = _pool()
    with pytest.raises(KeyboardInterrupt):
        with pool:
            raise KeyboardInterrupt
    _assert_reaped(pool)


def test_real_converter_initialization_failure_is_reported():
    """The production worker entry point must also propagate setup failures."""
    pool = _pool({"device": "invalid-device"}, startup_timeout=60)
    with pytest.raises(RuntimeError, match="worker"):
        with pool:
            pytest.fail("Invalid accelerator configuration should fail initialization")
    _assert_reaped(pool)


@pytest.fixture
def recovery_config(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    return {"mode": "memory_exit", "markers": str(markers), "calls": str(tmp_path / "calls.jsonl")}


@pytest.fixture
def started_processes(monkeypatch):
    """Retain both generations to verify complete process and pipe cleanup."""
    processes = []
    popen = pools.subprocess.Popen

    def start(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(pools.subprocess, "Popen", start)
    return processes


@pytest.mark.parametrize("mode", ["memory_exit", "killed"])
def test_memory_failure_retries_only_unfinished_work(
    worker_command, recovery_config, started_processes, tmp_path, mode
):
    """Acknowledged results stay complete, and the replacement pool is reused."""
    import json

    recovery_config["mode"] = mode
    with _pool(recovery_config) as pool:
        results = pool.convert([(Path(key), key) for key in ("done", "bad", "pending")], tmp_path)
        first = next(results)
        assert first[0] == Path("done")
        old_worker = pool.workers[0]
        remaining = list(results)
        assert [result[0] for result in remaining] == [Path("bad"), Path("pending")]
        assert all(result[1] for result in remaining)
        assert old_worker.process.poll() is not None and old_worker.connection.closed
        assert pool.workers[0].process is not old_worker.process
        assert list(pool.convert([(Path("next-batch"), "next-batch")], tmp_path))[0][1]
    calls = [json.loads(line)[1] for line in Path(recovery_config["calls"]).read_text().splitlines()]
    assert calls == ["done", "bad", "bad", "pending", "next-batch"]
    assert len(started_processes) == 2
    assert all(process.poll() is not None for process in started_processes)
    _assert_reaped(pool)


def test_repeated_memory_failure_is_fatal(worker_command, recovery_config, started_processes, tmp_path):
    recovery_config["always_fail"] = True
    pool = _pool(recovery_config)
    with pytest.raises(RuntimeError, match="bad.*single memory-recovery attempt"):
        with pool:
            list(pool.convert([(Path("bad"), "bad")], tmp_path))
    assert len(started_processes) == 2
    assert all(process.poll() is not None for process in started_processes)
    _assert_reaped(pool)


def test_recovery_budget_is_shared_across_batches(worker_command, recovery_config, started_processes, tmp_path):
    recovery_config["fail_keys"] = ["bad", "later"]
    pool = _pool(recovery_config)
    with pytest.raises(RuntimeError, match="later.*single memory-recovery attempt"):
        with pool:
            assert list(pool.convert([(Path("bad"), "bad")], tmp_path))[0][1]
            list(pool.convert([(Path("later"), "later")], tmp_path))
    assert len(started_processes) == 2
    assert all(process.poll() is not None for process in started_processes)
    _assert_reaped(pool)


def test_replacement_startup_failure_reaps_both_generations(
    worker_command, recovery_config, started_processes, tmp_path
):
    recovery_config["fail_restart"] = True
    pool = _pool(recovery_config)
    with pytest.raises(RuntimeError, match="worker"):
        with pool:
            list(pool.convert([(Path("bad"), "bad")], tmp_path))
    assert len(started_processes) == 2
    assert all(process.poll() is not None for process in started_processes)
    _assert_reaped(pool)


def test_worker_killed_before_submission_can_recover(worker_command, started_processes, tmp_path):
    import signal

    with _pool() as pool:
        old_worker = pool.workers[0]
        os.kill(old_worker.process.pid, signal.SIGKILL)
        old_worker.process.wait(timeout=5)
        assert list(pool.convert([(Path("input"), "input")], tmp_path))[0][1]
        assert old_worker.connection.closed
    assert len(started_processes) == 2
    assert all(process.poll() is not None for process in started_processes)
    _assert_reaped(pool)


def test_memory_recovery_reaps_all_workers_and_retries_interrupted_conversions(
    worker_command, recovery_config, started_processes, monkeypatch, tmp_path
):
    """Replacing one failed worker's pool must not lose another worker's task."""
    import json
    from collections import Counter

    recovery_config["wait_for"] = str(tmp_path / "interrupted-started")
    popen = pools.subprocess.Popen

    def start(*args, **kwargs):
        if len(started_processes) >= 2:
            assert all(process.poll() is not None for process in started_processes[:2])
        return popen(*args, **kwargs)

    monkeypatch.setattr(pools.subprocess, "Popen", start)
    with _pool(recovery_config, workers=2) as pool:
        old_workers = list(pool.workers)
        results = pool.convert([(Path(key), key) for key in ("done", "bad", "interrupted", "pending")], tmp_path)
        assert next(results)[0] == Path("done")
        remaining = list(results)
        assert {path for path, _, _ in remaining} == {Path("bad"), Path("interrupted"), Path("pending")}
        assert all(success for _, success, _ in remaining)
        assert all(worker.process.poll() is not None and worker.connection.closed for worker in old_workers)
    calls = [json.loads(line)[1] for line in Path(recovery_config["calls"]).read_text().splitlines()]
    assert Counter(calls) == {"done": 1, "bad": 2, "interrupted": 2, "pending": 1}
    assert len(started_processes) == 4
    assert all(process.poll() is not None for process in started_processes)
    _assert_reaped(pool)
