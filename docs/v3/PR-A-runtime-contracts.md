# V3-A: first integration PR (2026-10-29)

- Opt-in `rl_server.entrypoints.v3_train --spec FILE --dry-run`.
- Immutable, JSON-canonicalized `RunSpec` with SHA-256 identity; different environments/seeds/configurations have different digests.
- Explicit on/off-policy algorithm and action-space capabilities; multi-agent is deliberately unsupported.
- Auditable run lifecycle, checkpoint/final-evaluation guards.
- No default entrypoint, legacy Redis key, optimizer, or trained model is modified.
- This PR implements **contracts only**. Running v3 training requires PR-B/C, which are not in the upstream repository yet.
- The schema version is a contract version, **not** an automatic compatibility guarantee.
- Initial validation does not claim cryptographic authentication of build digests, only syntax/immutability.
- RunSpec ownership and durable uniqueness enforcement are **not** implemented; a production controller must reject duplicate run IDs.
- Real Gymnasium, MuJoCo, MicroRTS, Redis and GPU regression gates remain future work.

Local standalone V3 proof-of-concept is intentionally separate from this legacy-compatible PR.
