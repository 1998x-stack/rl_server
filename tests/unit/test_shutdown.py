"""Regression tests for graceful shutdown: bounded waits, one-shot steps, exit codes."""
import os
import signal
import time
from unittest.mock import MagicMock

import torch.multiprocessing as mp

from rl_server.entrypoints.train import _queue_depth, _run_shutdown, _stop_workers


def _stubborn_child(ready):
    """Ignore SIGTERM and never exit on its own: stop() must fall back to kill()."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(0.1)


class _StubbornWorker:
    """Minimal worker exposing what ``_stop_workers`` needs."""

    def __init__(self, process, log):
        self.process = process
        self.log = log

    def stop(self, timeout: float = 15.0):
        """Same contract as the real workers: terminate, join, then kill."""
        self.process.terminate()
        self.process.join(timeout)
        if self.process.is_alive():
            self.log.warning(f"did not exit gracefully pid={self.process.pid}, killing")
            self.process.kill()
            self.process.join(5)
        self.log.log_info(f"stopped exitcode={self.process.exitcode}")


def test_run_shutdown_executes_every_step_once_even_when_one_fails():
    log = MagicMock()
    calls = []

    def failing_step():
        calls.append("fail")
        raise BrokenPipeError("[Errno 32] Broken pipe")

    _run_shutdown((
        ("first", lambda: calls.append("first")),
        ("failing", failing_step),
        ("last", lambda: calls.append("last")),
    ), log)

    assert calls == ["first", "fail", "last"], "a failing step must not stop the shutdown sequence"
    log.exception.assert_called_once()
    assert "failing" in log.exception.call_args.args[0]


def test_stop_workers_kills_process_that_ignores_sigterm():
    mp.set_start_method("spawn", force=True)
    log = MagicMock()
    ready = mp.Event()
    process = mp.Process(target=_stubborn_child, args=(ready,), name="stubborn")
    process.start()
    ready.wait(30)

    started = time.time()
    _stop_workers([_StubbornWorker(process, log)], log, timeout=1.0)
    elapsed = time.time() - started

    assert not process.is_alive()
    assert elapsed < 20, "stop() must be bounded, not join forever"
    log.warning.assert_called()  # the kill fallback must be visible in the log


def test_stop_workers_survives_individual_failures():
    log = MagicMock()
    broken = MagicMock()
    broken.stop.side_effect = RuntimeError("boom")
    healthy = MagicMock()

    _stop_workers([broken, healthy, None], log)

    healthy.stop.assert_called_once()
    log.exception.assert_called_once()
    assert "RuntimeError" not in log.exception.call_args.args[0]  # message carries the worker type


def test_queue_depth_is_diagnostic_not_fatal():
    assert _queue_depth(MagicMock(qsize=lambda: 7)) == "7"

    exploding = MagicMock()
    exploding.qsize.side_effect = OSError("unavailable")
    assert _queue_depth(exploding) == "?"


def test_dead_workers_are_reported_once_and_counted():
    from unittest.mock import MagicMock, PropertyMock

    from rl_server.entrypoints.train import _report_dead_workers

    log = MagicMock()
    alive = MagicMock()
    alive.process = MagicMock()
    alive.process.is_alive.return_value = True

    dead = MagicMock()
    dead.process = MagicMock()
    dead.process.is_alive.return_value = False
    dead.process.pid = 4242
    dead.process.exitcode = 1

    reported = {}
    assert _report_dead_workers([alive, dead], log, reported) == 1
    log.error.assert_called_once()
    assert "exitcode=1" in log.error.call_args.args[0]

    log.reset_mock()
    assert _report_dead_workers([alive, dead], log, reported) == 1
    log.error.assert_not_called(), "a worker must not be re-reported every loop iteration"


def test_arrival_lag_and_reason_hints():
    from rl_server.entrypoints.train import (
        _arrival_lag_p95,
        _effective_lag,
        _reason_hint,
    )

    lags = [0, 0, 1, 1, 2, 2, 3, 3, 4, 40]
    assert _arrival_lag_p95([]) == "?"
    assert _arrival_lag_p95(lags) == "40", "p95 of 10 samples picks the largest"
    assert _effective_lag("auto", lags, cap=8) == 8, "auto must respect the cap"
    assert _effective_lag("auto", [0, 1, 1], cap=64) == 1
    assert _effective_lag("auto", [], cap=64) == 1
    assert _effective_lag(0, lags, cap=64) == 0, "explicit values must not be overridden"
    assert "grads" in _reason_hint("none_gradient")
    assert "NaN" in _reason_hint("non_finite")


def test_backlog_warning_fires_only_above_quarter_capacity():
    from unittest.mock import MagicMock

    from rl_server.entrypoints.train import _check_backlog

    log = MagicMock()
    _check_backlog(10, 1000, log, num_trainers=1)
    log.warning_throttled.assert_not_called()

    _check_backlog(400, 1000, log, num_trainers=1)
    message = log.warning_throttled.call_args.args[1]
    assert "backlog depth=400/1000" in message
    assert "num_trainers" in message

    log.reset_mock()
    _check_backlog(5, 0, log, num_trainers=1)
    log.warning_throttled.assert_not_called()
