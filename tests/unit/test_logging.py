"""Logging regression tests: spawn visibility, level plumbing, throttling, exception folding."""
import os
import subprocess
import sys
import textwrap

import pytest

from rl_server.utils.logging import Log, setup_logging


def _read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


@pytest.fixture()
def tmp_logs(tmp_path):
    return tmp_path


def test_file_keeps_debug_even_when_console_level_is_info(tmp_logs, capsys):
    log = Log("unit_level", level="INFO", base_dir=str(tmp_logs))
    log.debug("debug-only-line")
    log.info("info-line")
    log.close()

    content = _read(tmp_logs / "unit_level" / "unit_level.log")
    assert "debug-only-line" in content, "file handler must record DEBUG regardless of console level"
    assert "info-line" in content
    assert "debug-only-line" not in capsys.readouterr().out


def test_same_logger_level_changes_do_not_silently_disable_file(tmp_logs):
    log = Log("unit_level_switch", level="DEBUG", base_dir=str(tmp_logs))
    log.debug("first-debug")
    log.close()

    log2 = Log("unit_level_switch", level="WARNING", base_dir=str(tmp_logs))
    log2.debug("second-debug")
    log2.close()

    content = _read(tmp_logs / "unit_level_switch" / "unit_level_switch.log")
    assert "first-debug" in content
    assert "second-debug" in content
    assert log2.logger.propagate is False


def test_child_process_logs_are_not_dropped(tmp_logs):
    """Regression: spawn children used to inherit no handlers, so worker logs vanished."""
    import torch.multiprocessing as mp

    script = tmp_logs / "child_logging_probe.py"
    script.write_text(textwrap.dedent(f"""
        import torch.multiprocessing as mp
        from rl_server.utils.logging import Log


        def child(log):
            log.info("CHILD-INFO-VISIBLE")
            log.error("CHILD-ERROR-VISIBLE")


        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            log = Log("unit_child", base_dir={str(tmp_logs)!r})
            log.info("PARENT-INFO-VISIBLE")
            process = mp.Process(target=child, args=(log,))
            process.start()
            process.join()
            log.close()
    """), encoding='utf-8')

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=repo_root)
    subprocess.run([sys.executable, str(script)], cwd=repo_root, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    log_dir = tmp_logs / "unit_child"
    combined = "".join(_read(path) for path in sorted(log_dir.glob("*.log")))
    assert "PARENT-INFO-VISIBLE" in combined
    assert "CHILD-INFO-VISIBLE" in combined, "child process INFO log must reach a file"
    assert "CHILD-ERROR-VISIBLE" in combined
    # Child must not append to the parent's rotating file: it gets its own pid suffix.
    assert len(list(log_dir.glob("*.log"))) == 2


def test_info_throttled_reports_suppressed_counts(tmp_logs):
    log = Log("unit_throttle", base_dir=str(tmp_logs), throttle_secs=3600)
    emitted = [log.info_throttled("empty_queue", f"empty attempt={i}") for i in range(5)]
    log.info_throttled("empty_queue", "empty attempt=last", interval=0)
    log.close()

    assert emitted == [True, False, False, False, False]
    content = _read(tmp_logs / "unit_throttle" / "unit_throttle.log")
    assert content.count("empty attempt=") == 2
    assert "suppressed=4" in content


def test_info_every_and_once(tmp_logs):
    log = Log("unit_every", base_dir=str(tmp_logs))
    for i in range(6):
        log.info_every("progress", 3, f"progress step={i}")
    log.once("banner", "banner-line")
    log.once("banner", "banner-line")
    log.close()

    content = _read(tmp_logs / "unit_every" / "unit_every.log")
    assert content.count("progress step=") == 2  # every 3rd of 6
    assert content.count("banner-line") == 1


def test_exception_folding_keeps_counts_and_drops_traceback_flood(tmp_logs):
    log = Log("unit_exc", base_dir=str(tmp_logs), exc_report_secs=3600)

    for _ in range(50):
        try:
            raise BrokenPipeError("[Errno 32] Broken pipe")
        except BrokenPipeError:
            log.exception("train main loop")
    log.summary({"version": 7})
    log.close()

    content = _read(tmp_logs / "unit_exc" / "unit_exc.log")
    tracebacks = content.count("Traceback (most recent call last)")
    assert tracebacks == 1, "only the first occurrence may dump a full traceback"
    assert "type=BrokenPipeError" in content
    assert "count=50" in content, "summary must preserve the total occurrence count"
    assert "version=7" in content


def test_exception_key_separates_call_sites_but_merges_varying_ids(tmp_logs):
    log = Log("unit_exc_key", base_dir=str(tmp_logs), exc_report_secs=3600)

    def raiser(message):
        raise ValueError(message)

    for index in range(3):
        try:
            raiser("timeout 1234")
        except ValueError:
            log.exception("first site")
    for index in range(3):
        try:
            raiser(f"timeout {index}")
        except ValueError:
            log.exception("first site")
    log.close()

    content = _read(tmp_logs / "unit_exc_key" / "unit_exc_key.log")
    assert content.count("Traceback (most recent call last)") == 1
    assert "timeout #" in content, "digit-normalized values must collapse into one key"


def test_from_config_honours_logging_section(tmp_logs):
    config = {"logging": {"level": "DEBUG", "dir": str(tmp_logs), "throttle_secs": 5}}
    log = Log.from_config(config, "unit_config_log")
    log.debug("configured-debug")
    log.close()

    assert log.throttle_secs == 5.0
    assert "configured-debug" in _read(tmp_logs / "unit_config_log" / "unit_config_log.log")


def test_summary_reports_throttled_totals(tmp_logs):
    log = Log("unit_summary", base_dir=str(tmp_logs), throttle_secs=3600)
    for i in range(4):
        log.info_throttled("noise", "noise line")
    log.summary({"versions": 3})
    log.close()

    content = _read(tmp_logs / "unit_summary" / "unit_summary.log")
    assert "log summary duration=" in content
    assert "versions=3" in content
    assert "suppressed=3" in content


def test_checkpoint_failures_reach_the_log_file(tmp_logs, monkeypatch):
    """Checkpoint save/load failures used to be printed to stdout only."""
    import torch

    from rl_server.utils import checkpoint

    log = Log("unit_checkpoint", base_dir=str(tmp_logs))
    monkeypatch.setattr(checkpoint, "_checkpoint_log", log)

    broken = tmp_logs / "models" / "broken_x" / "broken_x_1_1.td"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a torch checkpoint")

    assert checkpoint.load_model(torch.nn.Linear(2, 2), "broken_x", "1", base_dir=str(tmp_logs / "models")) is None
    log.close()

    content = _read(tmp_logs / "unit_checkpoint" / "unit_checkpoint.log")
    assert "checkpoint load failed" in content
    assert "broken_x_1_1.td" in content


def test_masked_entropy_keeps_backward_finite(tmp_logs):
    """All-masked rows used to produce 0/0 entropy: forward hid it, backward returned NaN."""
    import torch

    # 其它测试会用 stub 顶掉 gym_microrts；先摘掉 stub，确保测的是真包。
    for name in [n for n in sys.modules if n.startswith('gym_microrts')]:
        if not hasattr(sys.modules[name], '__path__'):
            del sys.modules[name]
    pytest.importorskip("gym_microrts.envs.vec_env")
    from rl_server.algorithms.ppo.microrts import MaskedCategorical

    logits = torch.randn(4, 100, requires_grad=True)
    masks = torch.zeros(4, 100, dtype=torch.bool)
    masks[0, :5] = True          # partially masked row
    masks[2, 7] = True           # single legal action
    # rows 1 and 3 stay fully masked

    distribution = MaskedCategorical(logits=logits, masks=masks)
    entropy = distribution.entropy()
    assert torch.isfinite(entropy).all(), "entropy must be finite for fully masked rows"

    entropy.sum().backward()
    assert torch.isfinite(logits.grad).all(), "gradient must stay finite (was NaN before the fix)"
