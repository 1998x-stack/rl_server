#!/usr/bin/env bash
# Reproducible local environment bootstrap for rl_server (no sudo required).
#
#   bash scripts/bootstrap_env.sh            # base + dev deps only
#   bash scripts/bootstrap_env.sh --mujoco   # + gymnasium[mujoco]
#   bash scripts/bootstrap_env.sh --microrts # + MicroRTS stack (portable JDK 17)
#   bash scripts/bootstrap_env.sh --atari    # + ALE (ale-py ships ROMs) and OpenCV
#   bash scripts/bootstrap_env.sh --all
#
# Everything lands in ./.venv (gitignored) and ~/.local/opt for the JDK.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-$REPO_ROOT/.venv}"
PY_VERSION="${PY_VERSION:-3.11}"
JDK_DIR="${JDK_DIR:-$HOME/.local/opt/jdk17}"

WANT_MUJOCO=0
WANT_MICRORTS=0
WANT_ATARI=0
for arg in "$@"; do
    case "$arg" in
        --mujoco) WANT_MUJOCO=1 ;;
        --microrts) WANT_MICRORTS=1 ;;
        --atari) WANT_ATARI=1 ;;
        --all) WANT_MUJOCO=1; WANT_MICRORTS=1; WANT_ATARI=1 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

command -v uv >/dev/null || { echo "uv not found: pip install uv  (or https://docs.astral.sh/uv/)" >&2; exit 1; }
UV=(uv pip install --python "$VENV/bin/python")

echo "==> [1/4] venv: $VENV (python $PY_VERSION)"
[ -x "$VENV/bin/python" ] || uv venv --python "$PY_VERSION" "$VENV"

echo "==> [2/4] PyTorch CPU wheel + requirements"
# CPU index avoids pulling ~3 GB of CUDA wheels on machines without a GPU.
"${UV[@]}" --index-url https://download.pytorch.org/whl/cpu "torch>=2.10.0"
"${UV[@]}" -r requirements-dev.txt

if [ "$WANT_MUJOCO" = 1 ]; then
    echo "==> [3/4] MuJoCo"
    "${UV[@]}" "gymnasium[mujoco]"
fi

if [ "$WANT_ATARI" = 1 ]; then
    echo "==> Atari (ALE) + OpenCV"
    # ale-py >=0.10 bundles the ROMs, so AutoROM is not needed.
    "${UV[@]}" "gymnasium[atari]" opencv-python-headless
fi

if [ "$WANT_MICRORTS" = 1 ]; then
    echo "==> [4/4] MicroRTS (gym-microrts 0.3.2 + portable Temurin JDK 17)"

    if [ ! -x "$JDK_DIR/bin/java" ]; then
        mkdir -p "$(dirname "$JDK_DIR")"
        tmp="$(mktemp -d)"
        url="https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
        echo "    downloading JDK 17 (~185 MB, resumable)"
        for _ in 1 2 3 4 5; do
            curl -sSL --retry 5 --retry-all-errors -C - -o "$tmp/jdk.tar.gz" "$url" && break
            echo "    retrying download ($(stat -c%s "$tmp/jdk.tar.gz" 2>/dev/null || echo 0) bytes)"
            sleep 3
        done
        tar xzf "$tmp/jdk.tar.gz" -C "$tmp"
        mv "$tmp"/jdk-17* "$JDK_DIR"
        rm -rf "$tmp"
        mkdir -p "$HOME/.local/bin"
        ln -sfn "$JDK_DIR/bin/java" "$HOME/.local/bin/java"
        ln -sfn "$JDK_DIR/bin/javac" "$HOME/.local/bin/javac"
    fi
    echo "    java: $("$JDK_DIR/bin/java" -version 2>&1 | head -1)"

    # gym 0.23.1 is the last release exposing gym.envs.registry.env_specs,
    # which gym_microrts 0.3.2 imports at module load.
    "${UV[@]}" "gym-microrts==0.3.2" "gym==0.23.1" dacite

    # numpy 2.0 removed bare np.int / np.float aliases that gym_microrts uses,
    # including inside spawned subprocesses, so patch the installed source.
    "$VENV/bin/python" - <<'PY'
import os
import re

import gym_microrts

pattern = re.compile(r'\bnp\.(int|float|bool|object)\b(?!\d)')
patched = []
for root, _, files in os.walk(os.path.dirname(gym_microrts.__file__)):
    for name in files:
        if not name.endswith('.py'):
            continue
        path = os.path.join(root, name)
        source = open(path).read()
        fixed = pattern.sub(lambda m: m.group(1), source)
        if fixed != source:
            open(path, 'w').write(fixed)
            patched.append(os.path.relpath(path, os.path.dirname(gym_microrts.__file__)))
print("    numpy-compat patched:", *patched, sep="\n      ") if patched else print("    numpy-compat: already clean")
PY

    echo "    MicroRTS needs JAVA_HOME for JPype:"
    echo "      export JAVA_HOME=$JDK_DIR"
fi

echo
echo "==> done. activate with: source \"$VENV/bin/activate\""
echo "    smoke test:          python -m pytest tests -q"
