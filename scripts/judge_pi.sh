#!/usr/bin/env bash
# LLM judge backend for scripts/evaluate.py.
#
# prompt 从 stdin 进入，判决 JSON 从 stdout 出来。默认用 `pi -p`（本机已登录的 coding
# assistant CLI），也可通过 RL_JUDGE_CLI 换成任意「读 stdin / 写 stdout」的模型 CLI：
#
#   RL_JUDGE_CMD="bash scripts/judge_pi.sh"  python scripts/evaluate.py --run-dir .checks/<id>
#   RL_JUDGE_CLI="codex exec --full-auto"     bash scripts/judge_pi.sh < prompt.md
#
# 退出码：0 表示裁判成功产出内容（判决解析由评测器负责）；非 0 表示后端不可用。
set -uo pipefail

CLI="${RL_JUDGE_CLI:-pi -p --mode text}"
TIMEOUT_SECS="${RL_JUDGE_TIMEOUT:-300}"
SYSTEM_PROMPT_FILE="$(dirname "${BASH_SOURCE[0]}")/../eval/judge_system_prompt.txt"

PROMPT="$(cat)"
if [ -z "${PROMPT//[[:space:]]/}" ]; then
    echo "judge_pi: empty prompt on stdin" >&2
    exit 2
fi

ARGS=()
if [ -f "$SYSTEM_PROMPT_FILE" ]; then
    ARGS+=(--append-system-prompt "$SYSTEM_PROMPT_FILE")
fi

# 只用一条命令、不做 shell 展开，避免 prompt 里的引号/中文被二次解释。
RAW="$(printf '%s' "$PROMPT" | timeout "$TIMEOUT_SECS" $CLI "${ARGS[@]}" 2>/tmp/judge_pi_stderr.log)"
STATUS=$?

if [ -z "${RAW//[[:space:]]/}" ]; then
    echo "judge_pi: CLI produced no output (status=$STATUS); see /tmp/judge_pi_stderr.log" >&2
    exit 3
fi

# pi 会在 stdout 混入终端控制序列（如 OSC 通知），剥掉后再交给评测器解析。
printf '%s\n' "$RAW" | sed -E 's/\x1b\][^\x07]*\x07//g; s/\x1b\[[0-9;?]*[a-zA-Z]//g'
