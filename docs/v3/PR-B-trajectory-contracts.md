# PR-B: V3 trajectory boundary semantics

Independent of PR-A, additive modules under rl_server/core.

- Reward/value/GAE arrays always have [time, environment] shape. Old PPO flattened rollouts are **not** converted automatically.
- PPO enforces one policy version for the training window.
- Gym SAME_STEP must supply the actual final observation from info; the returned reset observation is never substituted into a Bellman target.
- Gym NEXT_STEP reset-only frames are skipped. DISABLED requires explicit reset.
- terminated stops bootstrap; truncated bootstraps from the true final observation but stops advantage propagation to a different episode.
- This PR tests synthetic adapter inputs and mathematical invariants. It does not instantiate a Gymnasium vector environment and is not a live algorithm convergence test.
- No legacy worker, Redis transport or algorithm code is modified; integration requires PR-C.
