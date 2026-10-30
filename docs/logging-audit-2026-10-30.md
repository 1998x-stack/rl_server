---
layout: default
title: Logging Audit (2026-10-30)
description: 日志系统审计 —— 信息过载与信息缺失的根因、修复与验证数据
---

# Logging Audit — 信息过载 / 信息缺失

审计对象：`rl_server/utils/logging.py` 及全部调用点（entrypoints / workers / transport）。
目标：找出“日志刷屏但不可用”与“关键信息根本没写下来”两类问题的**根因**，而不是调低日志级别。

## 1. 复现与量化

用一次受控的本地训练（DQN，40s，261 个版本）做基线与修复后对照：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 主进程日志行数 | **83,537 行 / 6.7 MB** | **41 行 / 8 KB** |
| 其中 `Shutting down train_main_local` | 64,936 行 | 1 行 |
| 其中完整 traceback | 1,084 份 | 0 份（折叠成 1 行摘要） |
| worker 进程（sampler/trainer/checker）日志 | **0 行**（全部丢失） | 每个进程独立文件，生命周期与进度齐全 |
| SIGTERM → 进程退出 | **永不退出**，只能 `kill -9` | 2.1s，exitcode=0，无残留进程 |
| 梯度批次被丢弃 | 128/249 = **51.4%** | 11/261 = 4.2%（仅启动预热） |

## 2. 根因

### 2.1 信息缺失：`spawn` 子进程的 handler 不会重建

`Log` 对象通过参数传进 `mp.Process`，`spawn` 出来的子进程里 `logging.getLogger(name).handlers == []`。
于是 `sampler/trainer/checker` 内部所有 `info`/`error` 记录被静默丢弃，只有 WARNING 以上会被
`logging.lastResort` 打到 stderr（无时间戳、无进程信息）。实测证据：

```
[child] handlers=[] level=0 propagate=True      # 子进程内
CHILD-INFO  → 丢失；CHILD-WARN → 仅 stderr
```

同一次故障还掩盖了第二个 bug：worker 退出时写 `self.process.pid`，而子进程里 `self.process` 是
`None`，退出日志必然 `AttributeError`——因为日志本来就看不见，所以一直没人发现。

### 2.2 信息缺失：`logger.setLevel(level)` 让"文件始终 DEBUG"变成假话

`setup_logging` 先把 logger 设为 `level`，文件 handler 的 DEBUG 设置因此永不生效（handler 级别
只能进一步收紧，不能放宽）。模块文档写着"文件始终 DEBUG"，实际文件里没有一行 DEBUG。

### 2.3 信息过载：关闭流程被写在热循环里，失败即无限重入

```python
while True:
    try:
        if should_exit():
            log.info("Shutting down ...")     # ← 每轮循环都打印
            model_dict['is_exit'] = True      # ← manager 已失效时抛 BrokenPipeError
            ...
            break
    except Exception:
        log.exception()                        # ← 吞掉异常后继续循环
```

SIGTERM 之后 manager 代理可能已失效，写 `is_exit` 抛 `BrokenPipeError` → `except` 捕获 → 循环继续
→ 再次打印 "Shutting down" → 再次抛异常。65,000 次空转，日志 6.7 MB，进程永不退出，CPU 100%。

### 2.4 信息过载：异常折叠的 key 设计错误

去重 key 是 `f"{type}:{str(exc_value)}"`：

- 消息里带 id/计数（`timeout 1234`）→ 每次都是新 key，永远去重不掉；
- 不同调用点消息相同 → 被错误合并，丢失调用点信息；
- 折叠时**只丢不报**，没有任何计数出口，无法回答"到底发生了多少次"。

### 2.5 信息过载：普通（非异常）消息没有任何限流手段

`pop_exps timeout x N`、`Version N`、关闭流程之类的高频 INFO 消息只能靠"每 N 次打印一次"的
手写取模，语义混乱（`rejected % 100 == 1`）、也没有"被抑制了多少条"的计数。心跳检查更糟：
它写在 `if version % 10 == 0:` 分支里，而该条件在整个版本窗口内恒为真——一旦 trainer 挂掉，
就是每秒上千条 `heartbeat stale`。

### 2.6 信息缺失：阻塞式 `put` 让 worker 无法响应退出，只能被强杀

`mp.Queue.put()` 不带超时时会永久阻塞。采样队列满（1000 条）且消费端已退出时，sampler 卡在
`put` 里，看不到 `is_exit`；父进程 `join(grace)` 等满 grace 时间后 `terminate()`。子进程随后还
会卡在解释器退出阶段的 queue feeder flush 上。实测：关闭耗时 10.0s，且 worker 的退出日志、
env 关闭（MicroRTS 的 JVM）全部丢失。

### 2.7 信息缺失 + 误导：梯度拒绝日志

`Rejected N stale or invalid gradient batches`（INFO，每 100 次一行）把两类完全不同的问题
（版本过期 / 数据非法）混成一个累计计数，既看不出原因，也看不出比例。实测埋点显示：

```
accepted=121 rejected=128
REJECTED lag(version-current) = {-1: 128}     # 100% 是"上一版本"的批次
```

根因是 `max_gradient_lag: 0` 与任何并发流水线都不相容：trainer 读取 `TRAIN_VERSION=V` 后开始
计算，主循环期间已经把版本推进到 `V+1`，于是这批梯度一出生就是"过期"的。严格的 `0` 只可能在
主循环被饿死时成立——等于一半 trainer 算力被白烧，而日志把它描述成"数据非法"。

### 2.8 信息缺失：错误与"没数据"不可区分

`pop_exps` / `pop_grads` 在任何异常下都 `return None`，与"队列为空"完全同形；调用方继续
`continue`，故障表现为"训练卡住"而不是"训练报错"。`RedisTrainer.run()` 甚至没有 try/except：
`run_once` 抛出即进程消失，日志里连一行都没有。

### 2.9 信息缺失：检查点保存失败只写在 stdout

`utils/checkpoint.py` 的 save/load 失败路径是 `print(f"Save model failed: ...")`——不进任何日志文件。
进程被重定向或跑在容器里时，丢检查点这件事就彻底静默了。同一类问题还有 microrts 里
每回合一次的 `print(outcomes ratio)`（stdout 刷屏），已改由 checker 的结构化日志输出。

## 3. 修复

| 位置 | 修复 |
|---|---|
| `utils/logging.py` | `Log.__getstate__/__setstate__` 在子进程重建 handler；logger 固定 DEBUG、console handler 受 `level` 约束；`propagate=False`；handler 签名变化时重建；子进程写 `{dir}.{pid}.log` |
| `utils/logging.py` | 异常 key 改为 `(类型, 最深栈帧, 数字归一化消息)`；重复只输出单行摘要（默认 60s）；`summary()` 在退出时输出每个异常的总次数与限流抑制数 |
| `utils/logging.py` | 新增 `info_throttled` / `warning_throttled` / `info_every` / `once`，输出均带 `suppressed=N` |
| `utils/logging.py` | 新增 `Log.from_config()`：`logging.level` / `logging.dir` / `logging.throttle_secs` 真正生效 |
| `entrypoints/train.py` | 关闭流程抽成一次性 `_shutdown(reason)`，每步单独兜底 → 无论哪步失败都 `break`；每步耗时 DEBUG + 超过 5s 告警 |
| `entrypoints/train.py` | 主循环从 `time.sleep(0)` 忙等改为 `grads_queue.get(timeout=0.05)`；心跳检查移到空闲分支并限流 30s |
| `entrypoints/train.py` | 启动一行写全配置（env/device/worker 数/队列/间隔/lag/prefix）；进度行带 `rate`、`accepted`、`rejected`、`queue_grads`；退出打印 `reason`、`elapsed`、`versions_per_sec` 与拒绝原因分项 |
| `workers/*.py` | 退出日志用 `os.getpid()`；`stop(grace, timeout)` 先给自愿退出窗口再 terminate，最后 kill；记录 `graceful=` 与 `exitcode=`；`put` 改带超时并在循环里检查 `is_exit`；退出前 `cancel_join_thread()` |
| `workers/{sampler,trainer,checker}.py` | 补 worker 级 telemetry（batches/s、权重同步次数、队列满等待、评估次数）并对崩溃路径写清阶段名 |
| `transport/redis_cache.py` | 重试改 WARNING + 无栈 + 限流，成功后补 `recovered` 行；空队列限流日志带 `queue_len`；有数据时输出 `resumed after N empty polls`；异常日志写清 op 名（不再与"空数据"同形） |
| `entrypoints/{sample,check,grads,trainer}.py` | 启动摘要（redis 拓扑、namespace、队列上限、聚合参数）；退出 summary；`RedisTrainer` 补 try/except 与进度；checker 每回合一行限流日志（原先每个指标一行） |
| `core/gradient_contract.py` | 新增 `gradient_batch_rejection()` 返回 `stale_version` / `future_version` / `shape_mismatch` / `non_finite` 等原因；`accepts_gradient_batch` 保留为布尔封装 |
| `utils/checkpoint.py` | save/load 失败从 `print` 改为写入 `logs/checkpoint/checkpoint.log`（带 path/prefix/version） |
| `config/default.yaml` | `max_gradient_lag: 1`（允许 1 个在途批次）、`version_log_interval: 10`、`logging.throttle_secs: 60` |

## 4. 验证

```bash
python -m pytest tests -q                 # 227 passed（含 tests/unit/test_logging.py、test_shutdown.py）
bash scripts/verify_lifecycle.sh local            # 本地：SIGTERM → ~2s 退出，exitcode=0
REDIS_PORT=6379 bash scripts/verify_lifecycle.sh distributed  # 分布式：三进程全部优雅退出
```

修复后的主进程日志（40s / 261 版本，全部 41 行）：

```
start env=DQNGymClassic device=cpu samplers=2 trainers=1 num_update_grads=1 max_gradient_lag=1 ...
start trainer processid 1993843 trainerid 0
gradient batch rejected reason=stale_version grads_version=0 current_version=2 lag=-2 total=1
version=260 rate=6.87ver/s accepted=260 rejected=11 queue_grads=0
checkpoint saved version=200 prefix=train_main_local_DQNGymClassic
shutdown requested reason=signal version=261 accepted=261 rejected=11
shutdown complete version=261 elapsed=38s versions_per_sec=6.82 accepted=261 rejected=11 reasons=[stale_version=11]
log summary duration=39s version=261 accepted=261 rejected=11 throttled_keys=2 suppressed=11
Exit OK
```


---

# 第二轮：逐环境实测（按日志读出来的问题）

一次性把 7 个算法注册项都跑过，边跑边读日志。每轮都修一个"日志能看出但代码没说清"的点。

## 逐环境结果（每项 15–45s，本地模式）

| 环境 | 版本数 | accepted/rejected | 由日志直接暴露的问题 |
|---|---|---|---|
| `DQNGymClassic` | 110 | 117 / 17 | 日志里只有速率，没有任何 loss 指标（见 3.1） |
| `MujocoNormal` | 60 | 63 / 65 | 每批梯度产出 ~26 条，`max_gradient_lag=1` 只能用到 2 条（50% 算力被丢） |
| `MujocoBeta` | 50 | 53 / 68 | 同上（56% 被丢） |
| `TD3MujocoNormal` | 60 | 69 / 32 | 同上，丢弃率 32% |
| `SACMujocoNormal` | 0 | 0 / 全部 | **100% 梯度 `none_gradient`：actor 参数永远没有梯度**（见 3.2） |
| `MicroRTS` | 230（修复后） | 238 / 2 | **修复前 0 个版本、208 条 NaN 梯度**（见 3.3） |
| `DQNAtari` | 0 | – | 缺少 `ale_py`，4 个 worker 全崩，主进程照跑 12s 并 `Exit OK`（见 3.4） |

## 本轮修掉的问题

### 3.1 训练指标完全没有出口

`TrainerWorker` 里 `grads_list, _ = calculate.generate_grads(...)`——第二个返回值被丢弃；PPO 把 loss
写进传进去的**临时 dict** `{'TRAIN_VERSION': ...}`，出了函数就没了；DQN/SAC/TD3 根本没有指标。
主进程里那段 `model_dict['trainer_metrics']` → TensorBoard 的代码因此永远不生效（PPO 也一样）。

修复：指标随梯度批次一起走（`grads_info['metrics']`），trainer 统一补上 `grad_norm`、`sample_version`，
算法写进 `calc_state['trainer_metrics']` 的数值键也会被收集；主进程逐版本写 TensorBoard 并把
`policy_loss/value_loss/entropy/gradient_norm/grad_norm/sample_lag` 打进进度行：

```
version=100 rate=5.60ver/s accepted=100 rejected=68 queue_grads=3 sample_lag=29 \
  policy_loss=20.56 value_loss=1619 entropy=-0.1863 gradient_norm=1897 grad_norm=1897
```

### 3.2 SAC 实际上什么都没训练

日志：`gradient rejection rate high rejected=20/20 (100%) reason=none_gradient`。
原因：`sac/mujoco_normal.py` 的 `generate_grads` 只对 `q_loss` 反传，策略网络（`mu`/`log_std`）与
熵温度没有任何 loss，于是这些参数的 `grad` 为 `None`，整批梯度被契约判为 `none_gradient`，版本号
永远停在 0。**这是算法实现缺失（缺 actor loss），不是日志问题**，本轮只保证它不再静默：
`reason=none_gradient` + `hint=algorithm produced params without grads` + 长时间无进展时 `stall` 中止。

### 3.3 MicroRTS：NaN 梯度（已修复）

修复前：`trainer progress ... grad_norm=nan`、`rejected ... reason=non_finite` 累计 208 条、0 个版本。
定位：`MaskedCategorical.entropy()` 用 `probs * masks` 后归一化，整行全掩时 `0/0` 得到 NaN；
`nansum` 让**前向**看起来是 0，但**反向**会把 NaN 传回 logits，实测 24 个参数里 22 个梯度是 NaN。

修复：熵改在 log 域计算（`log_softmax` + 把 -inf 显式置 0），并在 PPO ratio 前对被掩动作的
log_prob 做有限下界裁剪。验证：8 个并行环境的 512 步 rollout 全部 `non_finite=0`，实跑 45s 从
0 个版本变成 230 个版本、238 接受 / 2 拒绝。

### 3.4 worker 崩了但主进程"一切正常"

`DQNAtari` 缺少 `ale_py` 时，sampler×2 + checker 都以 `exitcode=1` 退出，主进程继续空转 12s，
最终日志仍是 `Exit OK`、进程 `exitcode=0`。

修复：

- 每个 worker 死亡立刻 `ERROR worker died worker=... pid=... exitcode=...`（只报一次）；
- 全部 worker 死亡 → `aborting: every worker exited` + 关闭 + `SystemExit(1)`；
- 新增停滞检测：`stall_warn_secs`（默认 60s）告警、`stall_fatal_secs`（默认 300s）中止并以 1 退出，
  用于覆盖"SAC 那样 worker 都活着但产不出可用梯度"的场景；
- trainer 心跳从未出现（初始化就崩）也计入不健康。

### 3.5 采样/评估不可中断导致退出慢 + semaphore 泄漏

sampler 一次 rollout 要 3–5s（MuJoCo）/ 数十秒（MicroRTS），期间看不到 `is_exit`，只能等满 grace 被
`terminate()` → 关闭耗时 5–10s，且被强杀的进程留下 `leaked semaphore objects` 警告、env/JVM 不关。

修复：`sample_multi_envs` 在每一步之间检查 `model_dict['is_exit']` 并 `return None`；`AlgoBaseAgent`
新增 `should_stop` 回调，`check_single_env` 在评估循环里中途返回 `None`（丢弃半回合），checker 据此
优雅退出。配合 `cancel_join_thread()` 与关闭 queue/manager，实测关闭 **2.1s**、无泄漏告警。

### 3.6 其它日志层面的小改进

- checker：由"每 60s 一条 + 只进 TensorBoard"改为"首评立即 + 每 `check_log_interval`(100) 次 + 每 60s"，
  退出时输出最后一次指标；浮点统一 4 位有效数字；评估耗时超过 30s 时输出 `check evaluating ...` 心跳。
- 启动行补 `tensorboard logdir=` 与 `models_dir=`（绝对路径）、`stall_warn/stall_fatal`；
  checkpoint 行改为输出实际文件路径，失败时 `ERROR checkpoint save failed`。
- 拒绝告警升级为可执行信息：`reason` + 拒绝率 + `grads_per_sample_batch` + `hint=raise max_gradient_lag to ~N`。

---


---

# 第三轮：按批准的三项改造

## 4.1 `max_gradient_lag` 改为自适应（`auto`）

原来的默认值把"流水线正常在途"的梯度当垃圾扔掉：trainer 读到 `TRAIN_VERSION=V` 后开始算，
主循环在它返回前已经推到 `V+k`，`lag` 小于 k 的批次必然过期——实测 MuJoCo 丢 50%、SAC 丢 90%。
手填一个常量既不可移植（算法/机器/worker 数一变就要重调），也不该让用户猜。

现在的策略：`max_gradient_lag: auto`（默认）+ `max_gradient_lag_cap: 64`。

- 主进程记录每条梯度**到达时的滞后** `current_version - grads_version`（滑动窗口 256 条）；
- 有效滞后取该分布的 **p95**（上限 `max_gradient_lag_cap`），即"正常在途"的规模；
- 真正的异常（积压到几百个版本、worker 挂掉后残留的旧批次）仍然被拒；
- 日志同时打印 `arrival_lag_p95=` 与 `max_lag=`，调参与否一目了然。

同类改造也应用到了 Redis 聚合端：`GradsAggregator` 的 `auto` 按一个聚合窗口
（`batch_update_grads_server` 条梯度）推导上限。

效果（同样的机器、同样的 18–22s 跑法）：

| 环境 | 改造前 accepted/rejected | 改造后 | 版本速率 |
|---|---|---|---|
| `DQNGymClassic` | 117 / 17 | **145 / 1** | 7.6 → 8.1 ver/s |
| `MujocoNormal` | 63 / 65 | **178 / 14** | 5.3 → 8.5 ver/s |
| `MujocoBeta` | 53 / 68 | **104 / 24** | 3.4 → 7.1 ver/s |
| `TD3MujocoNormal` | 69 / 32 | **115 / 13** | 4.0 → 7.6 ver/s |
| `MicroRTS` | 238 / 2 | **240 / 0** | 5.5 ver/s |
| `SACMujocoNormal` | 0 / 全部 | **131 / 61** | 0 → 6.2 ver/s |

## 4.2 SAC 补上 actor loss（此前策略根本没被训练）

`generate_grads` 只对 `q_loss` 反传，导致 `mu`/`log_std` 的 `grad` 恒为 `None`，整批梯度被契约
拒绝、版本停在 0。现在按标准 SAC 补全：

- critic：软贝尔曼误差，并且**终局转移不再自举**（原实现 `target = r + γ·min Q` 忽略了 `done`）；
- actor：重参数化策略梯度 `α·logπ − min(Q1,Q2)`，`tanh` 压缩后按 `log(1−tanh²)` 做雅可比修正
  （原来是直接对未压缩动作取 `log_prob`，策略梯度有偏）；
- actor 反传时冻结 critic 叶子参数，避免 critic 同时被两个方向推；
- 熵温度 α 固定 0.2（`TRAIN_CONFIG['ALPHA']`），并把 `q_loss/actor_loss/entropy/alpha`
  作为指标写回，日志与 TensorBoard 都能看到：

```
version=100 rate=5.89ver/s accepted=100 rejected=46 queue_grads=46 sample_lag=18 \
  arrival_lag_p95=31 max_lag=31 q_loss=0.9084 actor_loss=0.5721 entropy=2.33 alpha=0.2 grad_norm=2.974
```

`entropy` 随训练从 3.3 降到 2.3，`q_loss` 稳定在 0.6–1.0：策略确实在学（此前一条梯度都没有）。

## 4.3 Atari 环境可用性

`gymnasium[atari]` + `ale-py 0.12.1` 装好后 ALE 自带 ROM，不再需要 AutoROM；但 DQN 的帧预处理
还需要 OpenCV（`opencv-python-headless`），缺了它 4 个 worker 会立刻 `ModuleNotFoundError: cv2`。
这一轮正好验证了 3.4 的 worker 健康检查：缺依赖时日志立刻给出

```
[ERROR] worker died worker=SamplerWorker pid=... exitcode=1
```

而不是像以前那样静静跑 12s 再报 `Exit OK`。

---


---

# 第四轮：后续处理（SAC 升级 / 背压告警 / Atari 梯度裁剪）

## 5.1 SAC 升级为标准形态

| 项 | 之前 | 现在 |
|---|---|---|
| actor loss | **不存在**（策略参数无梯度，整批被拒） | `α·logπ − min(Q1,Q2)`，`tanh` 雅可比修正 |
| critic target | 用**在线** critic 自举 | 独立目标 critic + Polyak `τ=0.005` |
| 终局处理 | `target = r + γ·minQ`（忽略 `done`） | `target = r + γ(1−done)·minQ` |
| 熵温度 | 固定常量 | `log_alpha` 可学习，目标熵 `−act_dim` |
| 指标 | 无 | `q_loss / actor_loss / alpha_loss / entropy / alpha` 进日志与 TensorBoard |

55s 实跑：`version 0 → 511`、`alpha 0.2 → 0.187`、`entropy 3.30 → 2.25~2.42`、`q_loss 0.55 → 0.39~0.49`。

**checkpoint 兼容**：`log_alpha` 是新增参数，旧检查点会缺键。`load_model` 现在先严格加载，
失败则降级为 `strict=False` 并写一条 WARNING（带 `missing_keys`/`unexpected_keys`），
不再让整个训练起不来——测试 `tests/unit/test_checkpoint_compat.py` 覆盖。

## 5.2 梯度队列背压告警

生产快于消费时 `grads` 队列会一直涨，梯度也就越来越陈旧，而日志里只有一个裸的
`queue_grads=`。现在超过容量 25% 就限流告警，并给出可执行建议：

```
grads queue backlog depth=400/1000 (40%): trainers produce faster than the main loop applies;
consider raising training.num_trainers (now 1) or lowering num_update_grads
— otherwise stale batches accumulate
```

## 5.3 DQNAtari 梯度裁剪

Atari 像素输入的梯度范数实测到 **5.9e5**，聚合端会把多 worker 梯度直接相加，量级失控。
现在在计算端按全局 L2 范数裁剪（`TRAIN_CONFIG['MAX_GRAD_NORM']=10`，0 表示关闭），
并把裁剪前范数作为指标暴露：

```
version=10 ... q_loss=1585 grad_norm_preclip=5.923e+05 grad_norm=10
```

同一 20s 跑法中 `q_loss` 从 1585 → 560 → 298，说明裁剪后网络确实在学。

---

## 6. 遗留观察（未在本轮修复）

- sampler 只有在两次 rollout 之间才会检查 `is_exit`，因此关闭仍有 1–1.5s 的延迟；要更快需要
  在算法层把 `num_steps` 切成可中断的小段。
- 分布式模式下 `exps` 队列长期顶在 1000 上限（sampler 1303 批 / 287 次队列满等待），这是吞吐
  配比问题（1 trainer vs 1 sampler），不是日志问题；现在从日志里可以直接看出这一点。
- 5 处 `print("... check env is ...")`（各算法构造 checker agent 时打一次）仍是 stdout 输出，
  没有时间戳/进程信息。它们是单次信息，暂不改成 logger（算法层拿不到 `Log` 句柄）；
  若要统一，应把 `log` 注入 `AlgoBaseAgent` 构造参数。
