#!/usr/bin/env python3
"""独立评测器：复核检查产物、做回归检测，并可接一个 LLM judge。

与 ``scripts/run_checks.py`` 的关系是「生产者 / 消费者」：评测器**不导入 rl_server**，
也不复用运行器的判断逻辑——它自己重新解析 JUnit XML 与原始训练日志，因此运行器有 bug
时评测器不会跟着一起错。

判定层次：

1. **结构校验**：``summary.json`` 是否包含期望的套件与字段。
2. **复算**：从原始日志重新数版本数、错误、semaphore 泄漏等，而不是信运行器给的数字。
3. **回归检测**：与 ``eval/baseline.json`` 比较（测试数量、e2e 吞吐、lint/mypy 告警数），
   超出容差即失败。
4. **LLM judge（可选）**：把指标 + 日志片段 + 评分规则交给外部裁判，从语义上判断
   「日志/行为是否退化」。没有裁判后端时显式记为 ``skipped``，绝不静默通过。

用法::

    python scripts/evaluate.py --run-dir .checks/<id>
    python scripts/evaluate.py --run-dir .checks/<id> --update-baseline
    RL_JUDGE_CMD="my-judge" python scripts/evaluate.py --run-dir .checks/<id>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "eval" / "baseline.json"
EXPECTED_SUITES = ("unit", "lint", "typecheck", "e2e")

# 回归容差：e2e 吞吐允许下探到基线的 60%，测试数量允许 5% 波动（删测/新增都合理）。
RATE_TOLERANCE = 0.6
TEST_COUNT_TOLERANCE = 0.95
SHUTDOWN_LIMIT_S = 60.0

JUDGE_RUBRIC = """\
你是 RL 训练框架的评审员。下面给出一次自动化检查的产物（隔离运行器产出）。
请只依据给出的证据判断：
1. 是否有**行为退化**的迹象（日志量暴涨、指标变为 NaN、梯度拒绝率高、worker 崩溃、退出不干净）。
2. 是否有**可观测性缺口**（该有的指标/原因说明缺失，导致无法定位问题）。
3. 是否有**测试质量**问题（断言过弱、只验证不抛异常）。
输出严格的 JSON（不要 markdown 代码块）：
{"verdict": "pass" | "warn" | "fail",
 "regressions": ["..."],
 "observability_gaps": ["..."],
 "test_quality": ["..."],
 "must_fix": ["..."]}
"""


def _load_json(path: Path, default=None):
    """读 JSON，不存在时返回 default。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _parse_junit(path: Path) -> dict:
    """独立解析 JUnit XML：只依赖标准库，避免信任运行器的统计。"""
    result: dict = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "parse_error": None}
    if not path.exists():
        result["parse_error"] = f"missing {path}"
        return result
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        result["parse_error"] = f"invalid xml: {error}"
        return result
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        result["tests"] += int(suite.get("tests", 0))
        result["failures"] += int(suite.get("failures", 0))
        result["errors"] += int(suite.get("errors", 0))
        result["skipped"] += int(suite.get("skipped", 0))
    return result


def _recompute_e2e(log_path: Path) -> dict:
    """从原始日志重新计算 e2e 指标（不看运行器的数字）。"""
    metrics: dict = {
        "versions": 0, "accepted": 0, "rejected": 0, "versions_per_sec": 0.0,
        "errors": 0, "warnings": 0, "worker_deaths": 0, "leaked_semaphores": 0,
        "shutdown_complete": False, "nan_metrics": 0, "lines": 0,
    }
    if not log_path.exists():
        return metrics
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        metrics["lines"] += 1
        if "[ERROR]" in line:
            metrics["errors"] += 1
        if "[WARNING]" in line:
            metrics["warnings"] += 1
        if "worker died" in line:
            metrics["worker_deaths"] += 1
        if "leaked semaphore" in line:
            metrics["leaked_semaphores"] += 1
        if re.search(r"=\s*nan\b|=inf\b", line, re.IGNORECASE):
            metrics["nan_metrics"] += 1
        if "shutdown complete" in line:
            metrics["shutdown_complete"] = True
            for token in line.split():
                key, _, value = token.partition("=")
                if key in ("version", "accepted", "rejected"):
                    try:
                        metrics["versions" if key == "version" else key] = int(value)
                    except ValueError:
                        pass
                elif key == "versions_per_sec":
                    try:
                        metrics["versions_per_sec"] = float(value)
                    except ValueError:
                        pass
    return metrics


REJECT_REASON_PATTERN = re.compile(r"reasons=\[([^\]]*)\]")
REJECT_SINGLE_PATTERN = re.compile(r"gradient batch rejected reason=(\w+)")
WORKER_EXIT_PATTERN = re.compile(r"stopped (\w+) processid (\d+).*?graceful=(\w+) exitcode=(-?\d+)")

# 拒绝率阈值：超过 FAIL 判失败，超过 WARN 会作为观察项写进报告（不失败，避免短跑抖动）。
REJECT_PCT_FAIL = 50.0
REJECT_PCT_WARN = 25.0

METRIC_PATTERN = re.compile(
    r"\b(policy_loss|value_loss|q_loss|actor_loss|alpha_loss|entropy|alpha|"
    r"grad_norm_preclip|grad_norm|sample_lag|arrival_lag_p95|queue_grads|"
    r"reject_pct|clip_frac|grad_scale)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def _extract_rejections(text: str) -> dict:
    """提取拒绝率与按 reason 的直方图（含 shutdown 汇总行与逐条告警）。"""
    reasons: dict = {}
    accepted = rejected = 0
    for line in text.splitlines():
        match = REJECT_REASON_PATTERN.search(line)
        if match and "shutdown complete" in line:
            for item in match.group(1).split(","):
                name, _, value = item.partition("=")
                if name.strip():
                    try:
                        reasons[name.strip()] = int(value)
                    except ValueError:
                        pass
        if "[WARNING]" in line:
            single = REJECT_SINGLE_PATTERN.search(line)
            if single:
                reasons.setdefault(single.group(1), 0)
        token_match = re.search(r"accepted=(\d+) rejected=(\d+)", line)
        if token_match and "shutdown complete" in line:
            accepted, rejected = int(token_match.group(1)), int(token_match.group(2))
    total = accepted + rejected
    return {
        "accepted": accepted,
        "rejected": rejected,
        "reject_pct": round(100.0 * rejected / total, 1) if total else 0.0,
        "reasons": reasons,
    }


def _extract_worker_lifecycle(text: str) -> list[dict]:
    """从主日志抽取每个 worker 的 graceful/exitcode，覆盖"退出是否干净"维度。"""
    workers = []
    for line in text.splitlines():
        match = WORKER_EXIT_PATTERN.search(line)
        if match:
            workers.append({"worker": match.group(1), "pid": int(match.group(2)),
                            "graceful": match.group(3) == "True",
                            "exitcode": int(match.group(4))})
    return workers


def _extract_metrics(text: str) -> dict:
    """抽取指标的首/末值，用于趋势校验（每行可能同时出现多个指标）。"""
    trends: dict = {}
    for line in text.splitlines():
        for name, raw in METRIC_PATTERN.findall(line):
            try:
                value = float(raw)
            except ValueError:
                continue
            entry = trends.setdefault(name, {"first": value, "last": value, "samples": 0})
            entry["last"] = value
            entry["samples"] += 1
    return trends


# 趋势容差：只抓"明显发散"，避免短跑波动造成 flaky。
GRAD_NORM_GROWTH_LIMIT = 10.0
LOSS_GROWTH_LIMIT = 10.0
LAG_GROWTH_LIMIT = 20.0
LAG_ABSOLUTE_LIMIT = 200.0
SHUTDOWN_FAIL_S = 30.0
SHUTDOWN_WARN_S = 10.0
LOG_DENSITY_WARN = 50.0
ENTROPY_COLLAPSE_ABS = 1e-3
TREND_METRICS = ("grad_norm", "grad_norm_preclip")
LOSS_METRICS = ("policy_loss", "value_loss", "q_loss", "actor_loss")


def _check(suite: str, name: str, ok: bool, detail: str, warn: bool = False) -> dict:
    """构造一条检查记录。

    ``warn=True`` 表示"通过但有风险"：不判失败，但会进报告/裁判证据，
    避免像早期那样把 25–50% 的拒绝率悄悄放过。
    """
    return {"suite": suite, "check": name, "ok": ok, "detail": detail,
            "level": "warn" if (ok and warn) else ("fail" if not ok else "pass")}


def evaluate(run_dir: Path, baseline: dict) -> tuple[list[dict], dict]:
    """执行结构校验 + 复算 + 回归比对，返回 (检查列表, 观测摘要)。"""
    checks: list[dict] = []
    summary = _load_json(run_dir / "summary.json")
    observation: dict = {"run_dir": str(run_dir), "suites": {}, "e2e": []}

    if summary is None:
        checks.append(_check("summary", "summary.json parseable", False, "missing or invalid JSON"))
        return checks, observation

    present = {suite["suite"]: suite for suite in summary.get("suites", [])}
    for suite in EXPECTED_SUITES:
        checks.append(_check(suite, "suite present", suite in present,
                             "found" if suite in present else "suite did not run"))

    # --- unit ---
    if "unit" in present:
        junit = _parse_junit(run_dir / "unit" / "junit.xml")
        observation["suites"]["unit"] = junit
        checks.append(_check("unit", "junit parsed", junit["parse_error"] is None,
                             junit["parse_error"] or f"{junit['tests']} tests"))
        checks.append(_check("unit", "no failures",
                             junit["failures"] == 0 and junit["errors"] == 0,
                             f"failures={junit['failures']} errors={junit['errors']}"))
        base_tests = int(baseline.get("unit", {}).get("tests", 0))
        if base_tests:
            floor = base_tests * TEST_COUNT_TOLERANCE
            checks.append(_check("unit", "test count not regressed", junit["tests"] >= floor,
                                 f"{junit['tests']} >= {floor:.0f} (baseline {base_tests})"))

    # --- lint / typecheck ---
    for suite, key in (("lint", "findings"), ("typecheck", "errors")):
        if suite not in present:
            continue
        entry = present[suite]
        status = entry.get("status")
        observation["suites"][suite] = {"status": status, key: entry.get(key, 0)}
        if status == "skipped":
            checks.append(_check(suite, "tool available", False, entry.get("reason", "skipped")))
            continue
        count = int(entry.get(key, 0) or 0)
        allowed = int(baseline.get(suite, {}).get(key, 0))
        checks.append(_check(suite, f"no new {key}", count <= allowed,
                             f"{count} <= {allowed} (baseline)"))

    # --- e2e（用原始日志复算）---
    if "e2e" in present:
        for result in present["e2e"].get("results", []):
            env = result["env"]
            log_path = run_dir / "e2e" / env / "train_main_local.log"
            recomputed = _recompute_e2e(log_path)
            recomputed["env"] = env
            recomputed["exit_code"] = result.get("exit_code")
            recomputed["reported_versions"] = result.get("versions")
            recomputed["shutdown_s"] = result.get("shutdown_s")
            recomputed["runner_agrees"] = result.get("versions") == recomputed["versions"]
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            recomputed["start_lines"] = text.count("start env=")
            recomputed["log_for_env"] = f"start env={env} " in text
            recomputed["metrics"] = _extract_metrics(text)
            recomputed["rejections"] = _extract_rejections(text)
            recomputed["workers"] = _extract_worker_lifecycle(text)
            recomputed["log_lines_per_version"] = round(
                recomputed["lines"] / max(recomputed["versions"], 1), 2)
            observation["e2e"].append(recomputed)

            checks.append(_check(env, "exit code 0", result.get("exit_code") == 0,
                                 f"exit={result.get('exit_code')}"))
            checks.append(_check(env, "shutdown completed", recomputed["shutdown_complete"],
                                 "shutdown complete line present"))
            checks.append(_check(env, "trained versions > 0", recomputed["versions"] > 0,
                                 f"versions={recomputed['versions']}"))
            checks.append(_check(env, "no worker deaths", recomputed["worker_deaths"] == 0,
                                 f"deaths={recomputed['worker_deaths']}"))
            checks.append(_check(env, "no non-finite log values", recomputed["nan_metrics"] == 0,
                                 f"nan_lines={recomputed['nan_metrics']}"))
            # 除文本匹配外，对**解析出的指标值**再做一次 isfinite 校验（inf/overflow 也要抓）。
            bad_values = {
                name: values["last"] for name, values in recomputed["metrics"].items()
                if not math.isfinite(values["first"]) or not math.isfinite(values["last"])
            }
            checks.append(_check(env, "parsed metrics finite", not bad_values,
                                 f"non_finite={bad_values}" if bad_values else "all finite"))
            checks.append(_check(env, "no leaked semaphores", recomputed["leaked_semaphores"] == 0,
                                 f"leaks={recomputed['leaked_semaphores']}"))
            checks.append(_check(env, "runner metrics reproducible", recomputed["runner_agrees"],
                                 f"runner={result.get('versions')} evaluator={recomputed['versions']}"))
            # 日志归属：共享目录曾被复用，导致"这个环境的指标其实是别人的"。
            checks.append(_check(env, "log belongs to this env",
                                 recomputed["log_for_env"] and recomputed["start_lines"] == 1,
                                 f"start_lines={recomputed['start_lines']} matches={recomputed['log_for_env']}"))

            # 可观测性：训练遥测不能丢
            telemetry = [name for name in ("grad_norm", "policy_loss", "q_loss") if name in recomputed["metrics"]]
            checks.append(_check(env, "training telemetry present", bool(telemetry),
                                 f"metrics={sorted(recomputed['metrics'])}"))

            # 趋势：只抓明显发散
            for name in TREND_METRICS:
                entry = recomputed["metrics"].get(name)
                if not entry:
                    continue
                first, last = entry["first"], entry["last"]
                limit = max(abs(first or 0.0) * GRAD_NORM_GROWTH_LIMIT, 1.0)
                checks.append(_check(env, f"{name} not diverging", abs(last) <= limit,
                                     f"first={first:.4g} last={last:.4g} limit={limit:.4g}"))
            for name in LOSS_METRICS:
                entry = recomputed["metrics"].get(name)
                if not entry:
                    continue
                first, last = entry["first"], entry["last"]
                limit = max(abs(first or 0.0) * LOSS_GROWTH_LIMIT, 1.0)
                checks.append(_check(env, f"{name} not diverging", abs(last) <= limit,
                                     f"first={first:.4g} last={last:.4g} limit={limit:.4g}"))

            # 拒绝率：>FAIL 判失败；WARN~FAIL 判 warn（进报告与裁判证据）。
            rejections = recomputed["rejections"]
            if rejections["accepted"] or rejections["rejected"]:
                pct = rejections["reject_pct"]
                checks.append(_check(
                    env, "gradient rejection rate under control", pct <= REJECT_PCT_FAIL,
                    f"reject_pct={pct} (fail>{REJECT_PCT_FAIL}, warn>{REJECT_PCT_WARN}) "
                    f"reasons={rejections['reasons']}",
                    warn=pct > REJECT_PCT_WARN,
                ))

            # 滞后累积：sample_lag / arrival_lag_p95 不应随版本无界发散。
            for name, growth, absolute in (("sample_lag", LAG_GROWTH_LIMIT, LAG_ABSOLUTE_LIMIT),
                                           ("arrival_lag_p95", 5.0, 100.0)):
                entry = recomputed["metrics"].get(name)
                if not entry:
                    continue
                first, last = entry["first"], entry["last"]
                limit = max(abs(first) * growth, absolute)
                checks.append(_check(env, f"{name} bounded", abs(last) <= limit,
                                     f"first={first:.4g} last={last:.4g} limit={limit:.4g}",
                                     warn=abs(last) > limit * 0.5))

            # 梯度裁剪：post-clip 的 grad_norm=10 会掩盖原始尺度，单独看 clip 指标。
            clip_frac = recomputed["metrics"].get("clip_frac")
            grad_scale = recomputed["metrics"].get("grad_scale")
            if grad_scale:
                # 像素 DQN 起步阶段裁剪饱和是正常的；真正的问题是原始尺度失控。
                # fail: 超出裁剪上限 1e5 倍；warn: 超出 1e3 倍。
                checks.append(_check(env, "raw gradient scale sane", grad_scale["last"] <= 1e5,
                                     f"grad_scale(preclip/limit)={grad_scale['last']:.4g} "
                                     f"fail>1e5 warn>1e3",
                                     warn=grad_scale["last"] > 1e3))
                if clip_frac:
                    checks.append(_check(env, "gradient clipping not saturating",
                                         grad_scale["last"] <= 1e5,
                                         f"clip_frac={clip_frac['last']:.3f} "
                                         f"grad_scale={grad_scale['last']:.1f}",
                                         warn=clip_frac["last"] >= 0.9))

            # 熵坍缩：探索彻底消失通常意味着策略崩了。
            entropy = recomputed["metrics"].get("entropy")
            if entropy:
                checks.append(_check(env, "entropy not collapsed",
                                     abs(entropy["last"]) >= ENTROPY_COLLAPSE_ABS,
                                     f"first={entropy['first']:.4g} last={entropy['last']:.4g} "
                                     f"abs_min={ENTROPY_COLLAPSE_ABS}",
                                     warn=abs(entropy["last"]) < 0.05))

            # 关闭耗时与日志密度
            shutdown_s = recomputed.get("shutdown_s") or 0.0
            checks.append(_check(env, "shutdown time bounded", shutdown_s <= SHUTDOWN_FAIL_S,
                                 f"shutdown_s={shutdown_s}", warn=shutdown_s > SHUTDOWN_WARN_S))
            density = recomputed["log_lines_per_version"]
            checks.append(_check(env, "log density reasonable", density <= LOG_DENSITY_WARN,
                                 f"lines_per_version={density}", warn=density > LOG_DENSITY_WARN / 2))

            # worker 生命周期：每个 worker 都应有 graceful + exitcode 记录
            workers = recomputed["workers"]
            checks.append(_check(env, "worker exit codes recorded", bool(workers),
                                 f"workers={[(w['worker'], w['graceful'], w['exitcode']) for w in workers]}"))

            # 与基线比较：基线里有的指标不能消失（可观测性回归）
            base_metrics = set(baseline.get("e2e", {}).get(env, {}).get("metric_names", []))
            if base_metrics:
                missing = sorted(base_metrics - set(recomputed["metrics"]))
                checks.append(_check(env, "metrics not dropped vs baseline", not missing,
                                     f"missing={missing}" if missing else "all present"))

            base_rate = float(baseline.get("e2e", {}).get(env, {}).get("versions_per_sec", 0.0))
            if base_rate:
                floor = base_rate * RATE_TOLERANCE
                checks.append(_check(env, "throughput not regressed",
                                     recomputed["versions_per_sec"] >= floor,
                                     f"{recomputed['versions_per_sec']:.2f} >= {floor:.2f} ver/s"))
    return checks, observation


def build_baseline(observation: dict, previous: dict) -> dict:
    """用本次观测生成/更新基线（保留未观测到的旧条目）。"""
    baseline = json.loads(json.dumps(previous or {}))
    unit = observation.get("suites", {}).get("unit")
    if unit and not unit.get("parse_error"):
        baseline.setdefault("unit", {})["tests"] = unit["tests"]
    for suite, key in (("lint", "findings"), ("typecheck", "errors")):
        entry = observation.get("suites", {}).get(suite)
        if entry:
            baseline.setdefault(suite, {})[key] = entry.get(key, 0)
    for row in observation.get("e2e", []):
        entry = baseline.setdefault("e2e", {}).setdefault(row["env"], {
            "versions_per_sec": round(row["versions_per_sec"], 3),
            "versions": row["versions"],
        })
        entry["metric_names"] = sorted(row.get("metrics", {}))
        entry["reject_pct"] = row.get("rejections", {}).get("reject_pct", 0.0)
        entry["log_lines_per_version"] = row.get("log_lines_per_version", 0.0)
        entry["metric_trends"] = {
            name: {"first": values["first"], "last": values["last"]}
            for name, values in row.get("metrics", {}).items()
        }
        # 基线只升不降地记录吞吐，避免一次慢跑把门槛永久拉低。
        current = baseline["e2e"][row["env"]].get("versions_per_sec", 0.0)
        baseline["e2e"][row["env"]]["versions_per_sec"] = round(
            max(current, row["versions_per_sec"]), 3)
    return baseline


def build_judge_prompt(observation: dict, checks: list[dict], summary: dict) -> str:
    """拼出一份有界的评审 prompt（指标表 + 失败项 + 少量日志片段）。"""
    lines = [JUDGE_RUBRIC, "\n# 检查结果"]
    for row in observation.get("e2e", []):
        rejections = row.get("rejections", {})
        trends = row.get("metrics", {})
        trend_text = ", ".join(
            f"{name} {values['first']:.4g}->{values['last']:.4g}"
            for name, values in sorted(trends.items())[:6]
        ) or "none"
        lines.append(
            f"- {row['env']}: versions={row['versions']} rate={row['versions_per_sec']:.2f}/s "
            f"accepted={row['accepted']} rejected={row['rejected']} "
            f"reject_pct={rejections.get('reject_pct', 0)}% "
            f"reject_reasons={rejections.get('reasons', {})} "
            f"errors={row['errors']} warnings={row['warnings']} "
            f"log_lines={row['lines']} lines_per_version={row.get('log_lines_per_version')} "
            f"shutdown_s={row.get('shutdown_s')} "
            f"metrics[{trend_text}]"
        )
    failed = [c for c in checks if not c["ok"]]
    lines.append(f"\n# 未通过的检查（{len(failed)} 条）")
    lines.extend(f"- [{c['suite']}] {c['check']}: {c['detail']}" for c in failed[:20])
    # 逐环境汇总（避免长列表被截断导致裁判看不到某些环境的检查项）。
    lines.append("\n# 检查汇总（按环境）")
    by_suite: dict = {}
    for check in checks:
        bucket = by_suite.setdefault(check["suite"], {"pass": 0, "warn": 0, "fail": 0})
        bucket[check.get("level", "pass" if check["ok"] else "fail")] += 1
    for suite, counts in by_suite.items():
        lines.append(f"- {suite}: pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}")
    lines.append("\n# 每个环境实际执行并通过的关键检查（证据，不是忽略）")
    for suite in ("unit", "lint", "typecheck"):
        suite_checks = [c for c in checks if c["suite"] == suite]
        if suite_checks:
            lines.append(f"- {suite}: " + ", ".join(
                f"{c['check']}[{c.get('level')}]" for c in suite_checks[:12]))
    for env_name in [row["env"] for row in observation.get("e2e", [])]:
        env_checks = [c for c in checks if c["suite"] == env_name]
        lines.append(f"- {env_name}: " + ", ".join(
            f"{c['check']}[{c.get('level')}]" for c in env_checks[:14]))

    lines.append("\n# 非 pass 的检查明细")
    lines.extend(f"- [{c['suite']}] {c['check']}: {c.get('level')} ({c['detail']})"
                 for c in checks if c.get("level") in ("warn", "fail"))
    lines.append("\n# 套件状态")
    lines.append(json.dumps({s["suite"]: s.get("status") for s in summary.get("suites", [])},
                            ensure_ascii=False))
    # 警告样本：头尾各取一半，避免只看到启动阶段的预热告警（曾导致裁判误判
    # "grads_version 恒为 0"）。
    per_env = 6
    lines.append("\n# 警告/错误样本（每环境取首尾各 3 条）")
    for log in sorted(Path(observation["run_dir"]).glob("e2e/*/train_main_local.log")):
        env_name = log.parent.name
        entries = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                   if "[WARNING]" in line or "[ERROR]" in line]
        picked = entries[: per_env // 2] + entries[-(per_env // 2):]
        lines.append(f"## {env_name} ({len(entries)} warning/error lines)")
        lines.extend(line[:400] for line in picked)
    warned = [c for c in checks if c.get("level") == "warn"]
    lines.append(f"\n# warn 级检查（{len(warned)} 条，不判失败但需要关注）")
    lines.extend(f"- [{c['suite']}] {c['check']}: {c['detail']}" for c in warned[:20])
    return "\n".join(lines)


def run_judge(prompt: str) -> dict:
    """调用外部裁判：命令（stdin/stdout）或 OpenAI 兼容 HTTP 接口。"""
    command = os.environ.get("RL_JUDGE_CMD")
    if command:
        try:
            completed = subprocess.run(shlex.split(command), input=prompt, capture_output=True,
                                       text=True, timeout=600, check=False)
            return {"backend": f"cmd:{command}", "status": "ok",
                    "raw": completed.stdout.strip()[:4000],
                    "returncode": completed.returncode}
        except (OSError, subprocess.SubprocessError) as error:
            return {"backend": f"cmd:{command}", "status": "error", "error": str(error)}

    base = os.environ.get("RL_JUDGE_API_BASE")
    model = os.environ.get("RL_JUDGE_MODEL")
    if base and model:
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode()
        request = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('RL_JUDGE_API_KEY', '')}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode())
            content = body["choices"][0]["message"]["content"]
            return {"backend": f"api:{model}", "status": "ok", "raw": content[:4000]}
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as error:
            return {"backend": f"api:{model}", "status": "error", "error": str(error)}

    return {"backend": None, "status": "skipped",
            "reason": "no judge backend configured (set RL_JUDGE_CMD or RL_JUDGE_API_BASE/RL_JUDGE_MODEL)"}


ANSI_PATTERN = re.compile(r"\x1b\][^\x07]*\x07|\x1b\[[0-9;?]*[a-zA-Z]")


def parse_verdict(raw: str) -> dict:
    """从裁判输出里抽取 JSON 判决；抽不出来就退化为 warn + 原文。"""
    raw = ANSI_PATTERN.sub("", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0))
            if isinstance(verdict, dict) and "verdict" in verdict:
                return verdict
        except json.JSONDecodeError:
            pass
    return {"verdict": "warn", "unparsed": raw[:800]}


def main(argv: list[str] | None = None) -> int:
    """入口：评测一个运行目录，写报告，按需更新基线。"""
    parser = argparse.ArgumentParser(description="Independent evaluator")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--judge", default="auto", choices=("auto", "off"),
                        help="auto=有后端就跑，off=不跑")
    parser.add_argument("--strict-warn", action="store_true",
                        help="把 warn 也视为失败（CI 严格模式）")
    parser.add_argument("--judge-file", default=None,
                        help="读取预先计算好的裁判判决 JSON（CI 中由外部裁判生成）")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    baseline_path = Path(args.baseline)
    baseline = _load_json(baseline_path, default={}) or {}

    checks, observation = evaluate(run_dir, baseline)
    summary = _load_json(run_dir / "summary.json", default={}) or {}

    judge: dict = {"backend": None, "status": "skipped", "reason": "--judge off"}
    if args.judge == "auto":
        if args.judge_file:
            verdict = _load_json(Path(args.judge_file), default=None)
            judge = ({"backend": f"file:{args.judge_file}", "status": "ok", "verdict": verdict}
                     if verdict else {"backend": None, "status": "error",
                                      "error": f"unreadable {args.judge_file}"})
        else:
            prompt = build_judge_prompt(observation, checks, summary)
            (run_dir / "eval").mkdir(parents=True, exist_ok=True)
            (run_dir / "eval" / "judge_prompt.md").write_text(prompt, encoding="utf-8")
            judge = run_judge(prompt)
            if judge.get("status") == "ok":
                judge["verdict"] = parse_verdict(judge.get("raw", ""))

    failures = [c for c in checks if not c["ok"]]
    warnings_ = [c for c in checks if c.get("level") == "warn"]
    verdict = judge.get("verdict") or {}
    judge_failed = isinstance(verdict, dict) and verdict.get("verdict") == "fail"
    judge_warn = isinstance(verdict, dict) and verdict.get("verdict") == "warn"
    if failures or judge_failed:
        overall = "fail"
    elif warnings_ or judge_warn:
        overall = "warn"
    else:
        overall = "pass"

    report = {
        "run_dir": str(run_dir),
        "baseline": str(baseline_path),
        "overall": overall,
        "checks_total": len(checks),
        "checks_failed": len(failures),
        "checks_warned": len(warnings_),
        "failures": failures,
        "warnings": warnings_,
        "observation": observation,
        "judge": judge,
    }
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Evaluation report\n", f"- run: `{run_dir}`",
             f"- overall: **{overall}**",
             f"- checks: {len(checks) - len(failures)}/{len(checks)} passed, "
             f"{len(warnings_)} warn", ""]
    if failures:
        lines.append("## Failed checks\n")
        lines.extend(f"- **[{c['suite']}] {c['check']}** — {c['detail']}" for c in failures)
        lines.append("")
    if warnings_:
        lines.append("## Warnings (not failing, but surfaced to the judge)\n")
        lines.extend(f"- **[{c['suite']}] {c['check']}** — {c['detail']}" for c in warnings_)
        lines.append("")
    lines.append("## Per-environment (recomputed from raw logs)\n")
    lines.append("| env | versions | ver/s | accepted | rejected | errors | warn | lines |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in observation.get("e2e", []):
        lines.append(f"| {row['env']} | {row['versions']} | {row['versions_per_sec']:.2f} | "
                     f"{row['accepted']} | {row['rejected']} | {row['errors']} | "
                     f"{row['warnings']} | {row['lines']} |")
    lines.append("")
    lines.append(f"## LLM judge\n\n- status: `{judge.get('status')}`")
    if judge.get("status") == "ok" and isinstance(judge.get("verdict"), dict):
        for key in ("verdict", "regressions", "observability_gaps", "test_quality", "must_fix"):
            if key in judge["verdict"]:
                lines.append(f"- {key}: {judge['verdict'][key]}")
    elif judge.get("reason") or judge.get("error"):
        lines.append(f"- note: {judge.get('reason') or judge.get('error')}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.update_baseline:
        updated = build_baseline(observation, baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        print(f"[evaluate] baseline written to {baseline_path}")

    print(json.dumps({"overall": overall, "failed": len(failures), "warned": len(warnings_),
                      "report": str(out_dir / "report.md"),
                      "judge": judge.get("status")}, indent=2))
    # warn 不算失败（但会进报告与裁判证据）；--strict-warn 可把 warn 也当失败。
    if overall == "fail":
        return 1
    if overall == "warn" and args.strict_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
