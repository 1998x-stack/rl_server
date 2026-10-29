# RL Server production-readiness audit — 2026-10-29

**Base:** `a72aeb78404501cc5fa648e149757e40c64e4314`. **Patch series:** `hardening/redis-snapshot-safety-20261029`.

## Code-confirmed findings

1. **P0: Redis-wide destructive startup.** `GradsAggregator.__init__` called `clear_data()` on two Redis clients, which used `FLUSHALL`. This clears *every DB* on the backing Redis server, including unrelated workloads. The patch eliminates startup clearing and scopes manual cleanup to RL keys.
2. **P0: Split model publication.** Independent writes of `train_model` then `TRAIN_VERSION` could be observed inconsistently by samplers. Additionally `sample.py` updated its local model version even after `get_train_model()` returned False. The patch introduces a one-key snapshot and checks successful loading.
3. **P0: Unsafe object decoding.** Model data originally passed through `pickle.loads`. New model snapshots use `torch.load(..., weights_only=True)`. **Residual risk: experience and gradient envelopes still use pickle**, so Redis must be isolated and exclusively trusted.
4. **P1: Missing version validation.** Aggregator accepted any `grads_version` and gradient shape; the patch rejects invalid/stale gradients and nonfinite values. Lag defaults to zero, which prioritizes correctness for on-policy updates over throughput.
5. **P1: Broken standalone defaults.** `sample.py`, `grads.py`, `check.py` resolved a removed top-level `config/` directory. The patch points to `rl_server/config/default.yaml`.
6. **P1: Checkpoint trust boundary.** Explicit `weights_only=True` reduces arbitrary-object loading risk for checkpoints.
7. **P0: Incomplete operational model.** No durable Redis consumption/ACK, leader fencing, job identity, deployment liveness/readiness contract, benchmark-backed SLO, or isolated credential management.

## Implemented architecture change

Redis keys are scoped by `RL_RUN_ID` (same value in all workers of a job):

```text
<run-id>:train_snapshot   compressed torch archive: {version, state_dict}
<run-id>:TRAIN_VERSION    compatibility index
<run-id>:train_model      compatibility data
<run-id>:exps             legacy Redis list (NOT durable)
<run-id>:grads            legacy Redis list (NOT durable)
<run-id>:exit             job stop flag
```

The entire snapshot is loaded from one Redis GET; writes use a Redis transactional pipeline. The old pickle-only model key cannot be migrated automatically and is deliberately not unpickled. Re-publish model weights from a trusted local checkpoint.

### Deployment preconditions

- Set `RL_RUN_ID` to a **unique**, identical value for sampler/grads/trainer components of a training job. `rl_server_default` is for local development only; reusing it in a shared cluster may still collide.
- Do not expose Redis to untrusted networks. Use ACLs/TLS, secret injection and network policies; the default Compose setup is **not secure for production**.
- Do not call `clear_data()` during normal startup. It remains a scoped, destructive maintenance operation.
- Ensure GPU, MuJoCo and Redis dependencies are provisioned before using corresponding entrypoints. Tests with fakeredis are not proof of production Redis availability.
- Configure algorithm-specific gradient and sampling version contracts before treating convergence behavior as validated.
- Test upgrade compatibility: the snapshot has a new wire format and the current worker fleet must be upgraded together.

## Outstanding production gates

| Priority | Workstream | Measurable acceptance |
| --- | --- | --- |
| P0 | Replace arbitrary pickle messages with strict versioned tensor schema | Fuzz malformed/oversized payloads; no code execution |
| P0 | Durable processing with per-message identity, ACK and dedup | Network disconnect + worker death + restart without silent loss or unintended repeated update |
| P0 | Model publisher lease + fencing token | Split-brain test: two aggregators cannot publish conflicting versions |
| P0 | Bounded cancellation and lifecycle | SIGTERM against saturated queues exits within an agreed deadline |
| P0 | Full-state checkpoints | Resume restores optimizer state, RNG state, replay buffer and exact progress |
| P0 | Reproducible builds and artifact provenance | Lockfile, CI quality gates, SBOM, image digest, vulnerability scan |
| P1 | Algorithm-specific numerical parity | PPO/SAC/TD3/DQN multi-seed regressions, convergence baselines |
| P1 | Security isolation and deployment | Non-root image, TLS/ACL, credentials, per-job CPU/GPU/memory limits |
| P1 | Instrumentation, benchmark, tracing | Queue depth, parameter lag, invalid gradients, steps/sec, p95/p99 latency, recovery MTTR |
| P1 | CI/CD staged rollout, runbooks and DR | Canary, rollback, multi-host failure injection and published SLO |

**Verification status:** changes are submitted to a review branch. Build/test results must be read from CI or a real Python runtime after checkout. Do not reuse the old README test badge as evidence for this patch.

### Security advisory affecting earlier torch versions

PyTorch published GHSA-63cw-57p8-fm3p (2026-01-26): even `torch.load(weights_only=True)` may execute arbitrary code when handling crafted checkpoints in PyTorch <=2.9.1. Fixed in 2.10.0. The minimum dependency is therefore raised to `torch>=2.10.0`, and workers still must avoid untrusted checkpoints. (https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
