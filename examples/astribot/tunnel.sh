#!/usr/bin/env bash
# Run this on the robot PC (not the GPU). Forwards the GPU policy server
# (:8080) to a local port, without SSH multiplexing (avoids
# "ControlSocket already exists" / "Address already in use" from a stale master).
#
# Usage:
#   bash examples/astribot/tunnel.sh
#   LOCAL_PORT=8090 GPU=4_card_2 bash examples/astribot/tunnel.sh
#
# Then, in another terminal:
#   SERVER_ADDRESS=127.0.0.1:${LOCAL_PORT:-8090} bash examples/astribot/run_robot_client.sh
set -euo pipefail

GPU="${GPU:-4_card_2}"
LOCAL_PORT="${LOCAL_PORT:-8090}"
REMOTE_PORT="${REMOTE_PORT:-8080}"

# Drop a leftover multiplexed master so a new -L is not sent to the old socket.
ssh -O exit "$GPU" 2>/dev/null || true
rm -f "${HOME}/.ssh/control-"*"${GPU}"* 2>/dev/null || true

echo "Tunnel: localhost:${LOCAL_PORT} -> ${GPU}:127.0.0.1:${REMOTE_PORT}"
echo "Client: SERVER_ADDRESS=127.0.0.1:${LOCAL_PORT} bash examples/astribot/run_robot_client.sh"

exec ssh -N \
    -o ControlMaster=no \
    -o ControlPath=none \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$GPU"
