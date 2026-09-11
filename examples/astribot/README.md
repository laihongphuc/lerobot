# Astribot remote inference

Two ways to hide PI05 inference latency. Both drive the same SDK TCP bridge.

**A. Async inference (gRPC, split across machines).** GPU runs the policy, robot
PC runs the client and the bridge. The arm never talks to the GPU. Overlapping
chunks are blended by the client's `aggregate_fn`.

```text
Astribot S1  <-->  SDK TCP bridge :5555  <-->  robot_client     <-->  policy_server :8080
                   (robot PC)                    (robot PC)           (GPU)
```

**B. RTC rollout (single process).** `lerobot-rollout` runs the policy in-process
and re-plans each chunk against the actions still queued (real-time chunking), so
it must run on the GPU box and reach the bridge from there. See
[RTC rollout](#rtc-rollout-alternative-to-the-async-pair) below.

```text
Astribot S1  <-->  SDK TCP bridge :5555  <==ssh -R==>  lerobot-rollout --inference.type=rtc
                   (robot PC)                          (GPU)
```

Full contract, layout, units, and safety: [`REMOTE_ROBOT_INFERENCE.md`](../../REMOTE_ROBOT_INFERENCE.md).

## Start order

1. **GPU** — policy server (keep running):

   ```bash
   bash examples/astribot/run_policy_server.sh
   ```

2. **Robot PC** — SDK in Terminal / Power / Active / Home, cameras on, then the TCP bridge in **streaming** mode (30 Hz last-command loop, non-blocking action reads, obs without waiting for motion-done). See the runbook.

3. **Tunnel** (only if the robot LAN cannot reach the GPU). On the **robot PC**:

   ```bash
   bash examples/astribot/tunnel.sh
   ```

   That binds **local 8090** → GPU `127.0.0.1:8080` and ignores a stale SSH
   ControlSocket. Then start the client with the matching address:

   ```bash
   SERVER_ADDRESS=127.0.0.1:8090 bash examples/astribot/run_robot_client.sh
   ```

   If `8090` is also taken: `LOCAL_PORT=8091 bash examples/astribot/tunnel.sh`
   and set `SERVER_ADDRESS=127.0.0.1:8091`.

4. **Robot PC** — client. `CKPT` is a path **on the GPU** (the server loads it):

   ```bash
   CKPT=/path/on/gpu/to/checkpoint \
   TASK_INDEX=0 \
   bash examples/astribot/run_robot_client.sh
   ```

   The client loads [`tasks.yaml`](tasks.yaml) (the 6 coffee subtasks). Focus the
   client TTY: **1–6** jump, **n**/**c** next, **p** previous. Logs show
   `prompt[<hash>]=...` when the subtask changes.

   Single prompt instead of the list: `TASKS_FILE= TASK="your instruction" bash examples/astribot/run_robot_client.sh`.

   LoRA checkpoints (`adapter_config.json`) are loaded automatically on the server.

## RTC rollout (alternative to the async pair)

Real-time chunking keeps the policy one chunk ahead: while the arm executes the
queue, the next chunk is denoised with the leftover actions as a soft prefix, so
inference latency never shows up as a pause. It replaces both
`run_policy_server.sh` and `run_robot_client.sh`.

The policy runs in-process, so this runs on the **GPU**, and the bridge must be
reachable from there. On the **robot PC**, push port 5555 to the GPU:

```bash
ssh -N -R 5555:127.0.0.1:5555 4_card_2
```

That is a *reverse* forward, the opposite direction of `tunnel.sh`. Then on the
**GPU**:

```bash
CKPT=/home/phuclh12/Code/lerobot/outputs/pi05_mem_astribot_lora/checkpoints/last/pretrained_model \
TASK_INDEX=0 \
bash examples/astribot/run_rtc_rollout.sh
```

`TASK_INDEX` picks one line from [`tasks.yaml`](tasks.yaml); rollout takes a
single instruction, so use `INTERACTIVE=true` to retype it live with `/subtask`
(`/start`, `/reset`, `/stop` drive the run) instead of the client's 1–6 keys.

| var | default | meaning |
|---|---|---|
| `CKPT` | `./outputs/pi05_mem_astribot_lora/checkpoints/last/pretrained_model` | checkpoint path on this machine |
| `TASK` / `TASKS_FILE` / `TASK_INDEX` | `tasks.yaml`, `0` | instruction; `TASK` wins |
| `INTERACTIVE` | `false` | stdin session with `/subtask`, `/start`, `/stop` |
| `RTC_MODE` | `guided` | `trained` needs `--policy.rtc_training_max_delay > 0` at train time |
| `EXECUTION_HORIZON` | `10` | overlap re-planned each chunk; keep ≤ `chunk_size` (50) |
| `MAX_GUIDANCE_WEIGHT` | `10.0` | clamp on the RTC guidance term |
| `PREFIX_ATTENTION_SCHEDULE` | `EXP` | `ZEROS` / `ONES` / `LINEAR` / `EXP` |
| `QUEUE_THRESHOLD` | `30` | start the next chunk once the queue drops this low |
| `RETURN_TO_INITIAL` | `false` | move back to the startup pose on shutdown |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `127.0.0.1` / `5555` | SDK TCP bridge as seen from the GPU |
| `FPS` / `DURATION` / `DEVICE` | `30` / `0` / `cuda` | `DURATION=0` runs until Ctrl-C |

Relative-action checkpoints (this one) are re-anchored automatically before each
RTC chunk. LoRA adapters load the same way as on the policy server.

## Joints and cameras

Policy / dataset order is **trailing** 16-D:

`arm_left_j0..j6`, `arm_right_j0..j6`, `gripper_left`, `gripper_right`

Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist` (480×640). The wire uses `head` / `left_wrist` / `right_wrist` JPEGs; the robot class remaps both.

Arm joints are absolute radians. Grippers are **0–100 mm, 0 = open**. Never rescale 0–1 ↔ 0–100.
