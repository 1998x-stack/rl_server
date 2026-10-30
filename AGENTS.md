# Repository Guidelines

## Project Structure & Module Organization

`rl_server/` contains the Python package. Put algorithm implementations in `rl_server/algorithms/<name>/`, shared interfaces and buffers in `core/`, worker processes in `workers/`, CLI modules in `entrypoints/`, Redis communication in `transport/`, and YAML loading in `config/`. Tests live in `tests/unit/`, `tests/smoke/`, `tests/integration/`, and `tests/v3/`. Add user-facing explanations to `docs/`; `README.md` covers setup and common workflows. Generated checkpoints belong under `models/`, not in source control.

## Build, Test, and Development Commands

Use Python 3.11, as in CI. Install development dependencies with `python -m pip install -r requirements-dev.txt`. There is no separate build step; `python -m compileall -q rl_server tests` checks that Python files compile. Run the full regression suite with `python -m pytest tests -q`. For a focused run, use `python -m pytest tests/unit/test_dqn.py -v` or `python -m pytest tests/v3 -q`. Start local DQN training without Redis with `python -m rl_server.entrypoints.train --env-name DQNGymClassic`.

## Coding Style & Naming Conventions

Use four-space indentation and existing Python conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep algorithm-specific code in its algorithm package and use the lazy registry in `rl_server/algorithms/__init__.py` for new algorithms. No formatter, linter, or type checker is configured; follow nearby code and keep imports and public interfaces consistent.

## Testing Guidelines

Use `pytest` and name test files `test_*.py` and test functions `test_*`. Place isolated behavior checks in `tests/unit/`, model execution checks in `tests/smoke/`, and pipeline checks in `tests/integration/`. Redis tests use `fakeredis`, so the suite does not require a Redis server. `pytest.ini` applies a 30-second timeout per test. Add regression tests for behavior changes; no numeric coverage target is configured.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects with prefixes such as `feat:`, `fix:`, `test:`, `docs:`, and scoped variants like `fix(v3):`. Keep commits focused. In pull requests, describe the behavior changed, link any relevant issue, and report the test command and result. Include screenshots only when changing a visual surface.
