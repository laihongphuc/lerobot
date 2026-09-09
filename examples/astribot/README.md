# Astribot remote inference

GPU runs the policy. The robot PC runs the LeRobot client and the SDK bridge.
The arm never talks to the GPU.

```text
Astribot S1  <-->  SDK TCP bridge :5555  <-->  robot_client     <-->  policy_server :8080
                   (robot PC)                    (robot PC)           (GPU)
```

Full contract, layout, units, and safety: [`REMOTE_ROBOT_INFERENCE.md`](../../REMOTE_ROBOT_INFERENCE.md).

## Start order

1. **GPU** — policy server (keep running):

   ```bash
   bash examples/astribot/run_policy_server.sh
   ```

2. **Robot PC** — SDK in Terminal / Power / Active / Home, cameras on, then the TCP bridge in **streaming** mode (30 Hz last-command loop, non-blocking action reads, obs without waiting for motion-done). See the runbook.

3. **Tunnel** (only if the robot LAN cannot reach the GPU):

   ```bash
   ssh -N -L 8080:127.0.0.1:8080 user@gpu-box
   ```

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

## Joints and cameras

Policy / dataset order is **trailing** 16-D:

`arm_left_j0..j6`, `arm_right_j0..j6`, `gripper_left`, `gripper_right`

Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist` (480×640). The wire uses `head` / `left_wrist` / `right_wrist` JPEGs; the robot class remaps both.

Arm joints are absolute radians. Grippers are **0–100 mm, 0 = open**. Never rescale 0–1 ↔ 0–100.
