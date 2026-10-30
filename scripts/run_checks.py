#!/usr/bin/env python3
"""隔离的检查运行器：pytest / e2e / lint / typecheck。

设计要点：

* **隔离**：每次运行都新建 ``.checks/<run-id>/``，日志与模型目录通过
  ``RL_SERVER_LOG_ROOT`` / ``RL_SERVER_MODEL_DIR`` 重定向进去，绝不写工作区；
  每个套件在独立子进程里跑，互不共享环境变量或临时文件。
* **可审计**：每个套件输出原始日志 + ``summary.json``，供 ``scripts/evaluate.py``
  独立复核（评测器不导入 rl_server，只读这些产物）。
* **不静默跳过**：工具缺失时记 ``skipped`` 并写原因，评测器会把 skipped 当成显式状态。

用法::

    python scripts/run_checks.py all                # 全部套件
    python scripts/run_checks.py unit lint          # 指定套件
    python scripts/run_checks.py e2e --envs DQNGymClassic,MujocoNormal --seconds 15
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKS_ROOT = REPO_ROOT / ".checks"
TOOLING_VENV = REPO_ROOT / ".tooling"
SUITES = ("unit", "e2e", "lint", "typecheck")

# 环境可用性探测：(名字, 需要 import 的模块)
ENV_PROBES = {
    "DQNGymClassic": None,
    "MujocoNormal": "mujoco",
    "MujocoBeta": "mujoco",
    "SACMujocoNormal": "mujoco",
    "TD3MujocoNormal": "mujoco",
    "MicroRTS": "gym_microrts",
    "DQNAtari": "ale_py",
}


def _tool(name: str) -> str | None:
    """优先使用仓库内 `.tooling` 虚拟环境里的工具，其次 PATH。"""
    candidate = TOOLING_VENV / "bin" / name
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


def _run(cmd, cwd, env, log_path, timeout=None):
    """跑一条命令，把 stdout/stderr 落盘；返回 (returncode, 输出文本)。"""
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                cmd, cwd=str(cwd), env=env, stdout=handle,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
            handle.write(f"\n[run_checks] TIMEOUT after {timeout}s\n")
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return code, text, time.time() - started


def _base_env(run_dir: Path) -> dict:
    """隔离环境：产物目录、线程数、哈希种子全部固定。"""
    env = dict(os.environ)
    env.update({
        "RL_SERVER_LOG_ROOT": str(run_dir / "artifacts"),
        "RL_SERVER_MODEL_DIR": str(run_dir / "artifacts" / "models"),
        "RL_SERVER_TENSORBOARD_DIR": str(run_dir / "artifacts" / "tensorboard"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def run_unit(run_dir: Path, env: dict, timeout: int) -> dict:
    """pytest 单测/集成测试，产出 JUnit XML。"""
    out = run_dir / "unit"
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit.xml"
    code, text, elapsed = _run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--disable-warnings",
         f"--junitxml={junit}"],
        REPO_ROOT, env, out / "pytest.log", timeout=timeout,
    )
    tail = "\n".join(text.strip().splitlines()[-5:])
    return {
        "suite": "unit", "status": "passed" if code == 0 else "failed",
        "exit_code": code, "elapsed_s": round(elapsed, 1),
        "junit": str(junit) if junit.exists() else None,
        "tail": tail,
    }


def run_lint(run_dir: Path, env: dict, timeout: int) -> dict:
    """ruff 静态检查。"""
    out = run_dir / "lint"
    out.mkdir(parents=True, exist_ok=True)
    ruff = _tool("ruff")
    if not ruff:
        return {"suite": "lint", "status": "skipped", "reason": "ruff not installed",
                "hint": "uv pip install --python .tooling/bin/python ruff"}
    code, text, elapsed = _run([ruff, "check", "rl_server", "tests", "scripts",
                                "--output-format=concise"],
                               REPO_ROOT, env, out / "ruff.log", timeout=timeout)
    findings = [line for line in text.splitlines() if ": " in line and line.strip()]
    return {
        "suite": "lint", "status": "passed" if code == 0 else "failed",
        "exit_code": code, "elapsed_s": round(elapsed, 1),
        "findings": len([f for f in findings
                         if f.startswith(("rl_server/", "tests/", "scripts/"))]),
        "tail": "\n".join(text.strip().splitlines()[-5:]),
    }


def run_typecheck(run_dir: Path, env: dict, timeout: int) -> dict:
    """mypy 类型检查（配置见 mypy.ini）。"""
    out = run_dir / "typecheck"
    out.mkdir(parents=True, exist_ok=True)
    mypy = _tool("mypy")
    if not mypy:
        return {"suite": "typecheck", "status": "skipped", "reason": "mypy not installed",
                "hint": "uv pip install --python .tooling/bin/python mypy"}
    code, text, elapsed = _run([mypy], REPO_ROOT, env, out / "mypy.log", timeout=timeout)
    errors = [line for line in text.splitlines() if ": error:" in line]
    return {
        "suite": "typecheck", "status": "passed" if code == 0 else "failed",
        "exit_code": code, "elapsed_s": round(elapsed, 1),
        "errors": len(errors),
        "tail": "\n".join(text.strip().splitlines()[-5:]),
    }


def _available_envs(requested: list[str]) -> tuple[list[str], list[dict]]:
    """筛出真实可跑的环境，并记录被跳过的原因。"""
    available, skipped = [], []
    assert os.environ.get("PYTHONPATH") is not None or True  # 仅提示：cwd 已在仓库根
    for name in requested:
        probe = ENV_PROBES.get(name)
        if probe is None:
            available.append(name)
            continue
        check = subprocess.run(
            [sys.executable, "-c", f"import {probe}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if check.returncode == 0:
            available.append(name)
        else:
            skipped.append({"env": name, "reason": f"missing dependency: {probe}"})
    if "MicroRTS" in available and not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        available.remove("MicroRTS")
        skipped.append({"env": "MicroRTS", "reason": "no JDK on PATH (set JAVA_HOME)"})
    return available, skipped


def _parse_training_log(log_path: Path) -> dict:
    """从训练日志里抽取评测需要的指标（评测器会独立重算一遍）。"""
    metrics: dict = {"versions": 0, "accepted": 0, "rejected": 0, "errors": 0,
                     "warnings": 0, "worker_deaths": 0, "leaked_semaphores": 0,
                     "shutdown_complete": False, "versions_per_sec": 0.0}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[ERROR]" in line:
            metrics["errors"] += 1
        if "[WARNING]" in line:
            metrics["warnings"] += 1
        if "worker died" in line:
            metrics["worker_deaths"] += 1
        if "leaked semaphore" in line:
            metrics["leaked_semaphores"] += 1
        if "shutdown complete" in line:
            metrics["shutdown_complete"] = True
            for token in line.split():
                if token.startswith("version="):
                    metrics["versions"] = int(token.split("=", 1)[1])
                elif token.startswith("accepted="):
                    metrics["accepted"] = int(token.split("=", 1)[1])
                elif token.startswith("rejected="):
                    metrics["rejected"] = int(token.split("=", 1)[1])
                elif token.startswith("versions_per_sec="):
                    metrics["versions_per_sec"] = float(token.split("=", 1)[1])
    return metrics


def _run_one_env(env_name: str, run_dir: Path, base_env: dict, seconds: int) -> dict:
    """跑一个环境的短训练，SIGTERM 后收集退出码与日志指标。

    每个环境使用**独立的**日志/模型/TensorBoard 目录：共享目录会让后一个环境的日志
    追加到前一个后面，评测时读到别人的指标（曾真实发生）。
    """
    out = run_dir / "e2e" / env_name
    out.mkdir(parents=True, exist_ok=True)
    env = dict(base_env)
    env["RL_SERVER_LOG_ROOT"] = str(run_dir / "artifacts" / "e2e" / env_name)
    env["RL_SERVER_MODEL_DIR"] = str(run_dir / "artifacts" / "e2e" / env_name / "models")
    env["RL_SERVER_TENSORBOARD_DIR"] = str(run_dir / "artifacts" / "e2e" / env_name / "tensorboard")
    log_root = Path(env["RL_SERVER_LOG_ROOT"]) / "logs" / "train_main_local"
    stdout_path = out / "stdout.log"

    process = subprocess.Popen(
        [sys.executable, "-m", "rl_server.entrypoints.train", "--env-name", env_name],
        cwd=str(REPO_ROOT), env=env, stdout=open(stdout_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    time.sleep(seconds)
    started = time.time()
    graceful = True
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        graceful = False
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
    shutdown_seconds = round(time.time() - started, 2)

    main_log = log_root / "train_main_local.log"
    metrics = _parse_training_log(main_log)
    # 子进程日志单独成文件（train_main_local.<pid>.log），错误统计要把它们算进来。
    for child in sorted(log_root.glob("train_main_local.*.log")):
        child_metrics = _parse_training_log(child)
        metrics["errors"] += child_metrics["errors"]
        metrics["warnings"] += child_metrics["warnings"]
        shutil.copy2(child, out / child.name)
    if main_log.exists():
        shutil.copy2(main_log, out / "train_main_local.log")
    text = (out / "train_main_local.log").read_text(errors="replace") if (
        out / "train_main_local.log").exists() else ""
    return {
        "env": env_name, "exit_code": process.returncode, "graceful": graceful,
        "start_lines": text.count("start env="),
        "env_matches_log": f"start env={env_name} " in text,
        "shutdown_s": shutdown_seconds, "log_lines": (
            len((out / "train_main_local.log").read_text(errors="replace").splitlines())
            if (out / "train_main_local.log").exists() else 0),
        **metrics,
    }


def run_e2e(run_dir: Path, envs: list[str], seconds: int, timeout: int) -> dict:
    """端到端：逐环境短训 + 优雅退出验证。"""
    out = run_dir / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    env = _base_env(run_dir)
    available, skipped = _available_envs(envs)
    results = []
    for name in available:
        started = time.time()
        try:
            results.append(_run_one_env(name, run_dir, env, seconds))
        except Exception as error:  # noqa: BLE001 - 运行器必须把所有失败都落盘
            results.append({"env": name, "error": f"{type(error).__name__}: {error}"})
        finally:
            elapsed = time.time() - started
            if elapsed > timeout:
                results[-1]["timeout"] = True

    failed = [r for r in results if r.get("exit_code") != 0 or not r.get("shutdown_complete")
              or r.get("versions", 0) <= 0]
    return {
        "suite": "e2e", "status": "passed" if results and not failed else "failed",
        "results": results, "skipped_envs": skipped,
        "seconds_per_env": seconds,
    }


def main(argv: list[str] | None = None) -> int:
    """解析参数、顺序执行套件并写出 ``summary.json``。"""
    parser = argparse.ArgumentParser(description="Isolated check runner")
    parser.add_argument("suites", nargs="*", default=["all"], help="unit/e2e/lint/typecheck/all")
    parser.add_argument("--run-dir", default=None, help="产物目录（默认 .checks/<timestamp>）")
    parser.add_argument("--envs", default="auto", help="e2e 环境列表，auto=全部可用环境")
    parser.add_argument("--seconds", type=int, default=15, help="每个 e2e 环境跑多少秒")
    parser.add_argument("--timeout", type=int, default=1800, help="单元套件超时秒数")
    parser.add_argument("--evaluate", action="store_true",
                        help="跑完立即调用独立评测器（scripts/evaluate.py）")
    parser.add_argument("--update-baseline", action="store_true", help="评测时重建基线")
    parser.add_argument("--judge", default="auto", choices=("auto", "off"),
                        help="透传给评测器的 judge 开关")
    args = parser.parse_args(argv)

    requested = args.suites or ["all"]
    if "all" in requested:
        requested = list(SUITES)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir) if args.run_dir else CHECKS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env(run_dir)

    if args.envs == "auto":
        envs = list(ENV_PROBES)
    else:
        envs = [name.strip() for name in args.envs.split(",") if name.strip()]

    summaries = []
    for suite in requested:
        if suite == "unit":
            summaries.append(run_unit(run_dir, env, args.timeout))
        elif suite == "lint":
            summaries.append(run_lint(run_dir, env, args.timeout))
        elif suite == "typecheck":
            summaries.append(run_typecheck(run_dir, env, args.timeout))
        elif suite == "e2e":
            summaries.append(run_e2e(run_dir, envs, args.seconds, args.timeout))
        else:
            summaries.append({"suite": suite, "status": "failed", "reason": "unknown suite"})

    payload = {
        "run_id": run_id,
        "repo": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "suites": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir),
                      "status": {s["suite"]: s["status"] for s in summaries}}, indent=2))
    checks_ok = all(s["status"] in ("passed", "skipped") for s in summaries)

    if args.evaluate:
        evaluator = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"),
                     "--run-dir", str(run_dir), "--judge", args.judge]
        if args.update_baseline:
            evaluator.append("--update-baseline")
        return subprocess.run(evaluator, cwd=str(REPO_ROOT), check=False).returncode
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
