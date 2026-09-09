#!/usr/bin/env bash
# Robot PC: streams obs to the GPU policy server and applies action chunks
# to the local Astribot TCP bridge (default 127.0.0.1:5555).
#
# The checkpoint path is loaded *on the GPU*, so CKPT must exist there.
# If the robot LAN cannot reach the GPU, open a tunnel first:
#   ssh -N -L 8080:127.0.0.1:8080 user@gpu-box
# then leave SERVER_ADDRESS at 127.0.0.1:8080.
set -euo pipefail

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
CKPT="${CKPT:-./outputs/pi05_mem_astribot_lora/checkpoints/last/pretrained_model}"
# Subtask list (keys 1–6 / n / p). Set TASKS_FILE= to use a single --task string instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_FILE="${TASKS_FILE:-${SCRIPT_DIR}/tasks.yaml}"
TASK_INDEX="${TASK_INDEX:-0}"
TASK="${TASK:-}"
POLICY_TYPE="${POLICY_TYPE:-pi05}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
FPS="${FPS:-30}"
BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${BRIDGE_PORT:-5555}"
ROBOT_INTERLEAVED="${ROBOT_INTERLEAVED:-auto}"

cmd=(
    python -m lerobot.async_inference.robot_client
    --server_address="${SERVER_ADDRESS}"
    --robot.type=astribot
    --robot.host="${BRIDGE_HOST}"
    --robot.port="${BRIDGE_PORT}"
    --robot.robot_interleaved="${ROBOT_INTERLEAVED}"
    --policy_type="${POLICY_TYPE}"
    --pretrained_name_or_path="${CKPT}"
    --policy_device="${POLICY_DEVICE}"
    --client_device=cpu
    --actions_per_chunk="${ACTIONS_PER_CHUNK}"
    --fps="${FPS}"
    --task_index="${TASK_INDEX}"
)
if [[ -n "${TASKS_FILE}" ]]; then
    cmd+=(--tasks_file="${TASKS_FILE}")
fi
if [[ -n "${TASK}" ]]; then
    cmd+=(--task="${TASK}")
fi
exec "${cmd[@]}"
