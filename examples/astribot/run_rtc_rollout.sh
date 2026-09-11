#!/usr/bin/env bash
# Astribot RTC (real-time chunking) rollout — the alternative to the
# async_inference pair (run_policy_server.sh + run_robot_client.sh).
#
#   async_inference : policy_server (GPU) <--gRPC--> robot_client (robot PC) --> bridge
#   this script     : lerobot-rollout runs the policy IN-PROCESS and drives the
#                     bridge directly, hiding inference latency by re-planning
#                     each chunk against the actions still queued.
#
# Because the policy runs in-process, this must run on the GPU box. If the GPU
# is not the robot PC, push the bridge to the GPU first. On the robot PC:
#
#   ssh -N -R 5555:127.0.0.1:5555 4_card_2
#
# then leave BRIDGE_HOST/BRIDGE_PORT at 127.0.0.1:5555 below. Note this is a
# *reverse* forward (-R), the opposite direction of tunnel.sh, which forwards
# the GPU policy port out to the robot PC for the async client.
#
# Usage:
#   CKPT=/abs/path/on/gpu TASK_INDEX=0 bash examples/astribot/run_rtc_rollout.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:-./outputs/pi05_mem_astribot_lora/checkpoints/last/pretrained_model}"
DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-30}"
DURATION="${DURATION:-0}" # 0 = run until Ctrl-C
# Rollout takes a single instruction. Pick one of the 6 coffee subtasks by index,
# or set TASK directly. INTERACTIVE=true lets you retype it live with /subtask.
TASKS_FILE="${TASKS_FILE:-${SCRIPT_DIR}/tasks.yaml}"
TASK_INDEX="${TASK_INDEX:-0}"
TASK="${TASK:-}"
INTERACTIVE="${INTERACTIVE:-false}"

BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-5555}"
ROBOT_INTERLEAVED="${ROBOT_INTERLEAVED:-auto}"
# The robot moves back to its startup pose on shutdown when true. Off by default
# so Ctrl-C leaves the arms where they are.
RETURN_TO_INITIAL="${RETURN_TO_INITIAL:-false}"

# RTC. `guided` is inference-time Jacobian guidance and works with any pi0/pi05/
# smolvla checkpoint; `trained` needs one trained with --policy.rtc_training_max_delay > 0.
RTC_MODE="${RTC_MODE:-guided}"
# Overlap re-planned against the queued actions. Must stay <= chunk_size (50 here).
EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"
MAX_GUIDANCE_WEIGHT="${MAX_GUIDANCE_WEIGHT:-10.0}"
PREFIX_ATTENTION_SCHEDULE="${PREFIX_ATTENTION_SCHEDULE:-EXP}"
# Start the next chunk once the queue drops to this many actions.
QUEUE_THRESHOLD="${QUEUE_THRESHOLD:-30}"
RTC_DEBUG="${RTC_DEBUG:-false}"

if [[ -z "${TASK}" && -n "${TASKS_FILE}" ]]; then
    TASK="$(awk -v idx="${TASK_INDEX}" '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*$/ { next }
        {
            line = $0
            sub(/^[[:space:]]*-[[:space:]]*/, "", line)
            sub(/^"/, "", line)
            sub(/"$/, "", line)
            if (n == idx) { print line; exit }
            n++
        }' "${TASKS_FILE}")"
    if [[ -z "${TASK}" ]]; then
        echo "TASK_INDEX=${TASK_INDEX} not found in ${TASKS_FILE}" >&2
        exit 1
    fi
fi
if [[ -z "${TASK}" ]]; then
    echo "No task: set TASK=... or TASKS_FILE + TASK_INDEX" >&2
    exit 1
fi

echo "RTC rollout | ${RTC_MODE} | horizon=${EXECUTION_HORIZON} | bridge ${BRIDGE_HOST}:${BRIDGE_PORT}"
echo "Task: ${TASK}"

exec python -m lerobot.scripts.lerobot_rollout \
    --strategy.type=base \
    --policy.path="${CKPT}" \
    --device="${DEVICE}" \
    --robot.type=astribot \
    --robot.host="${BRIDGE_HOST}" \
    --robot.port="${BRIDGE_PORT}" \
    --robot.robot_interleaved="${ROBOT_INTERLEAVED}" \
    --inference.type=rtc \
    --inference.rtc.enabled=true \
    --inference.rtc.mode="${RTC_MODE}" \
    --inference.rtc.execution_horizon="${EXECUTION_HORIZON}" \
    --inference.rtc.max_guidance_weight="${MAX_GUIDANCE_WEIGHT}" \
    --inference.rtc.prefix_attention_schedule="${PREFIX_ATTENTION_SCHEDULE}" \
    --inference.rtc.debug="${RTC_DEBUG}" \
    --inference.queue_threshold="${QUEUE_THRESHOLD}" \
    --fps="${FPS}" \
    --duration="${DURATION}" \
    --task="${TASK}" \
    --interactive="${INTERACTIVE}" \
    --return_to_initial_position="${RETURN_TO_INITIAL}"
