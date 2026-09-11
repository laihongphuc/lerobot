#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Length-prefixed JSON TCP contract and trailing/interleaved joint layouts.

Wire images use short keys ``head`` / ``left_wrist`` / ``right_wrist``. Dataset
and policy keys are ``cam_head`` / ``cam_left_wrist`` / ``cam_right_wrist``.
"""

from __future__ import annotations

import base64
import json
import math
import socket
from typing import Any, Literal

import cv2
import numpy as np

Layout = Literal["trailing", "interleaved"]

N_JOINTS = 16
HEADER_SIZE = 4
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
# Joint angles live in radians (~[-π, π]); gripper opening is 0–100 mm.
_RADIAN_LIMIT = math.pi

TRAILING_JOINT_NAMES: tuple[str, ...] = (
    "arm_left_j0",
    "arm_left_j1",
    "arm_left_j2",
    "arm_left_j3",
    "arm_left_j4",
    "arm_left_j5",
    "arm_left_j6",
    "arm_right_j0",
    "arm_right_j1",
    "arm_right_j2",
    "arm_right_j3",
    "arm_right_j4",
    "arm_right_j5",
    "arm_right_j6",
    "gripper_left",
    "gripper_right",
)

INTERLEAVED_JOINT_NAMES: tuple[str, ...] = (
    *TRAILING_JOINT_NAMES[:7],
    TRAILING_JOINT_NAMES[14],
    *TRAILING_JOINT_NAMES[7:14],
    TRAILING_JOINT_NAMES[15],
)

CAMERA_KEYS: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
WIRE_IMAGE_KEYS: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
WIRE_TO_CAM: dict[str, str] = dict(zip(WIRE_IMAGE_KEYS, CAMERA_KEYS, strict=True))
CAM_TO_WIRE: dict[str, str] = {cam: wire for wire, cam in WIRE_TO_CAM.items()}

GRIPPER_NAME_TOKENS: tuple[str, ...] = ("gripper", "effector")


def is_gripper_like_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in GRIPPER_NAME_TOKENS)


def trailing_to_interleaved(values: list[float] | np.ndarray) -> list[float]:
    """``[L7, R7, gL, gR]`` → ``[L7, gL, R7, gR]``."""
    x = [float(v) for v in values]
    if len(x) != N_JOINTS:
        raise ValueError(f"Expected {N_JOINTS}-D values, got {len(x)}")
    return [*x[0:7], x[14], *x[7:14], x[15]]


def interleaved_to_trailing(values: list[float] | np.ndarray) -> list[float]:
    """``[L7, gL, R7, gR]`` → ``[L7, R7, gL, gR]``."""
    x = [float(v) for v in values]
    if len(x) != N_JOINTS:
        raise ValueError(f"Expected {N_JOINTS}-D values, got {len(x)}")
    return [*x[0:7], *x[8:15], x[7], x[15]]


def names_for_layout(layout: Layout) -> list[str]:
    if layout == "interleaved":
        return list(INTERLEAVED_JOINT_NAMES)
    return list(TRAILING_JOINT_NAMES)


def detect_layout(
    values: list[float] | np.ndarray | None,
    names: list[str] | None,
    flag: str = "auto",
) -> Layout:
    """Resolve robot-wire layout.

    Priority: explicit flag, then obs names, then ``qpos[7]`` units, else trailing.
    """
    flag = (flag or "auto").lower()
    if flag == "true":
        return "interleaved"
    if flag == "false":
        return "trailing"

    if names is not None and len(names) >= N_JOINTS:
        return "interleaved" if is_gripper_like_name(str(names[7])) else "trailing"

    if values is not None and len(values) >= 8:
        q7 = float(values[7])
        if q7 < 0.0:
            return "trailing"
        if q7 > _RADIAN_LIMIT:
            return "interleaved"

    return "trailing"


def extract_state(message: dict[str, Any]) -> tuple[list[float], list[str] | None]:
    """Pull 16-D values and optional names from an obs message."""
    state = message.get("state")
    names: list[str] | None = None
    values: Any = None
    if isinstance(state, dict):
        values = state.get("values")
        raw_names = state.get("names")
        if raw_names is not None:
            names = [str(n) for n in raw_names]
    if values is None and "qpos" in message:
        values = message["qpos"]
        raw_names = message.get("names")
        if raw_names is not None:
            names = [str(n) for n in raw_names]
    if values is None:
        raise ValueError("Obs missing 16-D state.values / qpos")
    values = [float(v) for v in values]
    if len(values) != N_JOINTS:
        raise ValueError(f"Bad qpos shape/values: ({len(values)},)")
    return values, names


def decode_jpeg_b64(data: str) -> np.ndarray | None:
    """Decode a base64 JPEG to an RGB uint8 ``(H, W, 3)`` array."""
    if not data:
        return None
    try:
        raw = base64.b64decode(data, validate=False)
    except (ValueError, TypeError):
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_jpeg_b64(image_rgb: np.ndarray, quality: int = 80) -> str:
    """Encode an RGB uint8 image as a base64 JPEG (for tests and debug dumps)."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def black_image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def decode_wire_images(
    message: dict[str, Any],
    height: int,
    width: int,
    fallback: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Map wire JPEG dict to ``cam_*`` RGB frames, filling gaps with fallback/black."""
    images = message.get("images") or {}
    frames: dict[str, np.ndarray] = {}
    for wire_key, cam_key in WIRE_TO_CAM.items():
        decoded = None
        payload = images.get(wire_key)
        if isinstance(payload, str):
            decoded = decode_jpeg_b64(payload)
        if decoded is None and fallback is not None:
            decoded = fallback.get(cam_key)
        if decoded is None:
            decoded = black_image(height, width)
        frames[cam_key] = decoded
    return frames


def parts_for_layout(values: list[float], layout: Layout) -> dict[str, list[float]]:
    if layout == "interleaved":
        return {
            "astribot_arm_left": values[0:7],
            "astribot_gripper_left": [values[7]],
            "astribot_arm_right": values[8:15],
            "astribot_gripper_right": [values[15]],
        }
    return {
        "astribot_arm_left": values[0:7],
        "astribot_arm_right": values[7:14],
        "astribot_gripper_left": [values[14]],
        "astribot_gripper_right": [values[15]],
    }


def build_action_message(
    trailing_values: list[float],
    layout: Layout,
    *,
    prompt: str | None = None,
    frame_index: int | None = None,
    seq: int | None = None,
) -> dict[str, Any]:
    """Build the JSON action the SDK bridge expects.

    ``trailing_values`` are policy/dataset order. They are remapped to ``layout``
    for the wire. Grippers are clipped to ``[0, 100]`` (mm), not rescaled.
    Echo ``seq`` when the bridge tags obs with a sequence number.
    """
    if len(trailing_values) != N_JOINTS:
        raise ValueError(f"Expected {N_JOINTS}-D trailing action, got {len(trailing_values)}")
    clipped = list(trailing_values)
    clipped[14] = min(100.0, max(0.0, clipped[14]))
    clipped[15] = min(100.0, max(0.0, clipped[15]))
    wire_values = trailing_to_interleaved(clipped) if layout == "interleaved" else clipped
    names = names_for_layout(layout)
    payload: dict[str, Any] = {
        "type": "action",
        "layout": layout,
        "values": wire_values,
        "names": names,
        "action": {"values": wire_values, "names": names},
        "parts": parts_for_layout(wire_values, layout),
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if frame_index is not None:
        payload["frame_index"] = frame_index
    if seq is not None:
        payload["seq"] = int(seq)
        payload.setdefault("frame_index", int(seq))
    return payload


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, HEADER_SIZE)
    length = int.from_bytes(header, "big")
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid message length {length}")
    payload = recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def send_message(sock: socket.socket, obj: dict[str, Any]) -> None:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(len(payload).to_bytes(HEADER_SIZE, "big") + payload)


def pack_message(obj: dict[str, Any]) -> bytes:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return len(payload).to_bytes(HEADER_SIZE, "big") + payload


def unpack_message(data: bytes) -> dict[str, Any]:
    if len(data) < HEADER_SIZE:
        raise ValueError("Truncated length header")
    length = int.from_bytes(data[:HEADER_SIZE], "big")
    payload = data[HEADER_SIZE : HEADER_SIZE + length]
    if len(payload) != length:
        raise ValueError(f"Truncated payload: expected {length}, got {len(payload)}")
    return json.loads(payload.decode("utf-8"))
