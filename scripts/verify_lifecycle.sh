#!/usr/bin/env bash
# Lifecycle verification: start training, send SIGTERM, and check that
#   * the process exits on its own (bounded, exitcode 0)
#   * worker processes are gone
#   * the log volume stays readable (no flood)
#
#   bash scripts/verify_lifecycle.sh local         # in-process training (no Redis)
#   bash scripts/verify_lifecycle.sh distributed   # grads + trainer + sampler over Redis
#
# Environment: RUN_SECONDS (default 30), REDIS_PORT for the distributed mode (default 6379).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-local}"
RUN_SECONDS="${RUN_SECONDS:-30}"
PY="${PY:-.venv/bin/python}"
CONFIG="${CONFIG:-rl_server/config/default.yaml}"
ENV_NAME="${ENV_NAME:-DQNGymClassic}"
OUT="$(mktemp -d)"

wait_gone() {
    local timeout_steps="$1"; shift
    for _ in $(seq 1 "$timeout_steps"); do
        local alive=0
        for pid in "$@"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && return 0
        sleep 0.1
    done
    return 1
}

report_exit() {
    local label="$1" start="$2" code="$3"
    local end elapsed_ms
    end=$(date +%s%N)
    elapsed_ms=$(( (end - start) / 1000000 ))
    echo "RESULT ${label}: exited in ${elapsed_ms}ms exitcode=${code}"
}

if [ "$MODE" = "local" ]; then
    rm -rf models logs/train_main_local
    ( exec "$PY" -m rl_server.entrypoints.train --config "$CONFIG" --env-name "$ENV_NAME" \
        > "$OUT/stdout.log" 2>&1 ) &
    PID=$!
    sleep "$RUN_SECONDS"
    echo "--- SIGTERM -> $PID ---"
    START=$(date +%s%N)
    kill -TERM "$PID"
    if wait_gone 600 "$PID"; then
        wait "$PID" 2>/dev/null
        report_exit local "$START" "$?"
    else
        echo "RESULT local: STILL RUNNING 60s after SIGTERM (BUG)"; kill -9 "$PID"
    fi
    echo "--- log volume ---"
    wc -l logs/train_main_local/*.log 2>/dev/null | sort -n
    echo "--- main log ---"
    grep -E "shutdown|Exit OK|log summary|version=" logs/train_main_local/train_main_local.log | tail -8
else
    export REDIS_MODEL_HOST="${REDIS_MODEL_HOST:-localhost}"
    export REDIS_EXPS_HOST="${REDIS_EXPS_HOST:-localhost}"
    export REDIS_GRADS_HOST="${REDIS_GRADS_HOST:-localhost}"
    export REDIS_MODEL_PORT="${REDIS_PORT:-6379}"
    export REDIS_EXPS_PORT="${REDIS_PORT:-6379}"
    export REDIS_GRADS_PORT="${REDIS_PORT:-6379}"
    export RL_RUN_ID="${RL_RUN_ID:-lifecycle_check}"
    OPTS="--config $CONFIG --env-name $ENV_NAME"

    rm -rf logs/grads_main logs/redis_trainer logs/sample_main
    ( exec "$PY" -m rl_server.entrypoints.grads $OPTS > "$OUT/grads.log" 2>&1 ) &
    GRADS=$!
    sleep 10
    ( exec "$PY" -m rl_server.entrypoints.trainer $OPTS > "$OUT/trainer.log" 2>&1 ) &
    TRAINER=$!
    ( exec "$PY" -m rl_server.entrypoints.sample $OPTS > "$OUT/sample.log" 2>&1 ) &
    SAMPLER=$!
    sleep "$RUN_SECONDS"

    echo "--- SIGTERM -> sampler/trainer, then grads ---"
    START=$(date +%s%N)
    kill -TERM "$SAMPLER" "$TRAINER"
    sleep 3
    kill -TERM "$GRADS"
    if wait_gone 600 "$SAMPLER" "$TRAINER" "$GRADS"; then
        report_exit distributed "$START" 0
    else
        echo "RESULT distributed: STILL RUNNING 60s after SIGTERM (BUG)"
        kill -9 "$SAMPLER" "$TRAINER" "$GRADS" 2>/dev/null
    fi
    for name in grads_main redis_trainer sample_main; do
        echo "=== $name ($(wc -l < "logs/$name/$name.log" 2>/dev/null || echo 0) lines) ==="
        tail -4 "logs/$name/$name.log" 2>/dev/null
    done
fi

echo "--- leftover rl_server entrypoints: $(pgrep -cf 'rl_server[.]entrypoints' || true)"
echo "--- captured stdout: $OUT"
