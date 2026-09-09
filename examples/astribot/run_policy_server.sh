#!/usr/bin/env bash
# GPU box: empty policy server. The robot client sends the checkpoint path
# on the first handshake (must be a path that exists on THIS machine).
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"

exec python -m lerobot.async_inference.policy_server \
    --host="${HOST}" \
    --port="${PORT}" \
    --fps="${FPS}"
