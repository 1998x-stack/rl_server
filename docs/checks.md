---
layout: default
title: Isolated Checks & Evaluation
description: 隔离检查运行器（pytest / e2e / lint / typecheck）与独立评测器（复核、回归检测、LLM judge）
---

# 隔离检查与独立评测

两个脚本，职责严格分开：

| 脚本 | 角色 | 是否导入 `rl_server` |
|---|---|---|
| `scripts/run_checks.py` | **生产者**：隔离地跑 unit / e2e / lint / typecheck，落盘原始日志 + `summary.json` | 否（只在子进程里调用） |
| `scripts/evaluate.py` | **消费者/评审**：重新解析产物、回归检测、可选 LLM judge | 否（只用标准库读文件） |

评测器不信任运行器的判断：版本数、错误、semaphore 泄漏等指标都由它**自己从原始训练日志重算**，
并额外检查「运行器报的数字」与「重算的数字」是否一致（不一致本身就是缺陷）。

## 快速开始

```bash
# 1) 跑检查（产物落在 .checks/<UTC 时间戳>/，不会写工作区）
python scripts/run_checks.py all                # unit + e2e + lint + typecheck
python scripts/run_checks.py unit lint          # 只跑部分
python scripts/run_checks.py e2e --envs DQNGymClassic,MujocoNormal --seconds 20

# 2) 独立复核
python scripts/evaluate.py --run-dir .checks/<id>
python scripts/evaluate.py --run-dir .checks/<id> --update-baseline   # 重建基线
```

## 隔离机制

| 维度 | 做法 |
|---|---|
| 日志 | `RL_SERVER_LOG_ROOT=<run>/artifacts`（相对 `logging.dir` 也解析到这里） |
| 模型 | `RL_SERVER_MODEL_DIR=<run>/artifacts/models` |
| TensorBoard | `RL_SERVER_TENSORBOARD_DIR=<run>/artifacts/tensorboard` |
| 线程/随机性 | `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTHONHASHSEED=0` |
| 进程 | 每个 e2e 用独立 session 启动，`SIGTERM` 后最多等 60s，超时才 `SIGKILL` |

`.checks/` 与 `.tooling/` 已加入 `.gitignore`。

## 判定模型：pass / warn / fail

| 级别 | 含义 | 是否失败 |
|---|---|---|
| `pass` | 达到预期 | 否 |
| `warn` | 通过但有风险（拒绝率 25–50%、样本滞后接近上限、裁剪饱和、熵偏低、关闭 >10s） | 否（报告 + 裁判证据里显式列出；`--strict-warn` 可升级为失败） |
| `fail` | 违反硬约束（非零退出、无版本推进、worker 崩溃、NaN/Inf、语义错乱、拒绝率 >50%、滞后 >20× 且 >200、grad_scale >1e5） | 是 |

`overall` 也随之三级化：任一 fail → fail；有 warn 或裁判给 warn → warn；否则 pass。

## 套件与判定

| 套件 | 工具 | 通过条件 |
|---|---|---|
| `unit` | `pytest tests -q --junitxml=...` | 无 failure/error，且用例数不低于基线 95% |
| `e2e` | 逐环境短训练 + `SIGTERM` | 退出码 0、`shutdown complete`、版本数 > 0、无 worker 死亡、无非有限指标、无 semaphore 泄漏、吞吐不低于基线 60% |
| `lint` | `ruff check rl_server tests` | 无新告警（与基线比较） |
| `typecheck` | `mypy`（见 `mypy.ini`） | 无新错误（与基线比较） |

环境可用性自动探测：缺少 `mujoco` / `gym_microrts`(+JDK) / `ale_py` 的环境会被记为 `skipped_envs`
并写明原因，而不是静默通过。

工具装在独立 `.tooling` venv 里，不污染运行时依赖：

```bash
uv venv --python 3.11 .tooling
uv pip install --python .tooling/bin/python ruff mypy
```

## 回归基线

`eval/baseline.json` 记录：用例数、lint 告警数、mypy 错误数、每个环境的 e2e 吞吐（`versions_per_sec`）。
吞吐基线「只升不降」：一次慢跑不会把门槛永久拉低。

```json
{
  "unit": {"tests": 238},
  "lint": {"findings": 0},
  "typecheck": {"errors": 0},
  "e2e": {"DQNGymClassic": {"versions_per_sec": 7.74, "versions": 87}}
}
```

## 内置的真实裁判：`scripts/judge_pi.sh`

默认后端是 `pi -p`（本机已登录的 coding assistant CLI），也可以用任意「读 stdin / 写 stdout」
的模型 CLI：

```bash
RL_JUDGE_CMD="bash scripts/judge_pi.sh" python scripts/evaluate.py --run-dir .checks/<id>
RL_JUDGE_CLI="codex exec --full-auto"  bash scripts/judge_pi.sh < prompt.md
```

脚本会剥掉 CLI 混入终端控制序列（OSC/ANSI），带 300s 超时，并在 `eval/judge_system_prompt.txt`
里给出评分口径（行为退化 / 可观测性缺口 / 测试质量）。裁判输出必须是 JSON：

```json
{"verdict": "pass|warn|fail", "regressions": [], "observability_gaps": [],
 "test_quality": [], "must_fix": []}
```

这个「裁判 → 反馈 → 修代码」的闭环在本仓库实际跑过：它先后指出
「拒绝率 44% 却判通过」「grads_version 恒为 0」「post-clip grad_norm 掩盖梯度爆炸」
「警告样本只看得到启动瞬态」等问题，对应的修复是：

| 裁判反馈 | 修复 |
|---|---|
| 拒绝率 warn 档未传导到判定 | 引入 pass/warn/fail 三级判定 + `--strict-warn` |
| 启动瞬态拒绝虚高拒绝率 | `cold_start_versions`（默认 20）内的拒绝单独计为 `cold_start_rejected` |
| 只看到启动期 `grads_version=0` | 裁判 prompt 改为每环境取首尾各 3 条告警 |
| `grad_norm=10` 掩盖原始尺度 | Atari 上报 `grad_norm_preclip` / `clip_frac` / `grad_scale`，并加阈值检查 |
| post-clip 恒等导致断言无效 | 阈值检查改看 `grad_scale`（原始倍数）而非裁剪后范数 |
| DQN/Beta/TD3 没有算法侧指标 | 三个 calculator 全部补 `q_loss` / `policy_loss` / `entropy` 等 |
| 18 条 hint 文案自指 | `max_gradient_lag: auto` 已生效时改为提示吞吐方向（加 trainer / 调 cap） |

## LLM judge

评测器可把「指标表 + 未通过检查 + 警告样本」组成有界 prompt，交给任意外部裁判：

```bash
# 方式一：命令行裁判（prompt 走 stdin，判决走 stdout）
RL_JUDGE_CMD="my-llm-cli --model x" python scripts/evaluate.py --run-dir .checks/<id>

# 方式二：OpenAI 兼容 HTTP 接口
RL_JUDGE_API_BASE=https://api.example.com/v1 RL_JUDGE_MODEL=some-model \
RL_JUDGE_API_KEY=... python scripts/evaluate.py --run-dir .checks/<id>

# 方式三：CI 中由外部裁判先生成判决文件
python scripts/evaluate.py --run-dir .checks/<id> --judge-file judge.json
```

裁判需输出严格 JSON：

```json
{"verdict": "pass|warn|fail", "regressions": [], "observability_gaps": [],
 "test_quality": [], "must_fix": []}
```

`verdict: fail` 会让整体判定失败、`warn` 会让整体降为 warn；没有配置裁判后端时状态是 `skipped`
（显式记录，不会伪装成通过）。
prompt 会写到 `<run>/eval/judge_prompt.md`，便于复现或人工评审。

## 产物结构

```
.checks/<id>/
├── summary.json            # 运行器输出（套件状态 + 原始指标）
├── unit/{junit.xml,pytest.log}
├── lint/ruff.log
├── typecheck/mypy.log
├── e2e/<env>/{stdout.log,train_main_local.log,train_main_local.<pid>.log}
├── artifacts/              # 隔离的 logs / models / tensorboard
└── eval/{report.json,report.md,judge_prompt.md}
```

## 端到端示例（本次改动后实测）

```
unit passed | e2e passed | lint passed | typecheck passed
DQNGymClassic    12s  versions=87  rate=7.74/s  errors=0  shutdown=1.3s
MujocoNormal     12s  versions=48  rate=4.33/s  errors=0  shutdown=1.4s
SACMujocoNormal  12s  versions=30  rate=2.73/s  errors=0  shutdown=1.2s
MicroRTS         12s  versions=23  rate=2.23/s  errors=0  shutdown=1.5s
DQNAtari         12s  versions=23  rate=2.10/s  errors=0  shutdown=1.4s

evaluate: overall=pass, checks 46/46 passed, judge=pass
```

回归检测演示：把某个 e2e 日志的 `shutdown complete` 删掉、注入 `leaked semaphore` 并把
`version=4x` 改成 `version=nan`，再在 JUnit 里塞一个 failure，评测器立刻报出 6 条失败
（含 `runner metrics reproducible` 不一致项），整体判定 `fail`。
