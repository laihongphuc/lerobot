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
"""Astribot TCP protocol helpers — no robot, GPU, or live socket required."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.robots.astribot import Astribot, AstribotConfig
from lerobot.robots.astribot.protocol import (
    CAMERA_KEYS,
    INTERLEAVED_JOINT_NAMES,
    N_JOINTS,
    TRAILING_JOINT_NAMES,
    WIRE_IMAGE_KEYS,
    build_action_message,
    decode_jpeg_b64,
    decode_wire_images,
    detect_layout,
    encode_jpeg_b64,
    extract_state,
    interleaved_to_trailing,
    pack_message,
    trailing_to_interleaved,
    unpack_message,
)
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features


def _trailing_sample() -> list[float]:
    # Dim 7 is arm_right_j0 (small radian); dim 14/15 are grippers in mm.
    values = [0.01 * i for i in range(14)]
    values[7] = -0.35
    values.extend([12.0, 80.0])
    return values


def _interleaved_sample() -> list[float]:
    return trailing_to_interleaved(_trailing_sample())


class TestLayoutRemap:
    def test_round_trip_trailing_interleaved(self):
        trailing = _trailing_sample()
        recovered = interleaved_to_trailing(trailing_to_interleaved(trailing))
        assert recovered == pytest.approx(trailing)

    def test_interleaved_puts_left_gripper_at_index_7(self):
        trailing = _trailing_sample()
        interleaved = trailing_to_interleaved(trailing)
        assert interleaved[7] == pytest.approx(trailing[14])
        assert interleaved[15] == pytest.approx(trailing[15])
        assert interleaved[8:15] == pytest.approx(trailing[7:14])

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="16-D"):
            trailing_to_interleaved([0.0] * 8)
        with pytest.raises(ValueError, match="16-D"):
            interleaved_to_trailing([0.0] * 15)


class TestDetectLayout:
    def test_names_index_7_gripper_is_interleaved(self):
        layout = detect_layout(_interleaved_sample(), list(INTERLEAVED_JOINT_NAMES), "auto")
        assert layout == "interleaved"

    def test_names_index_7_arm_is_trailing(self):
        layout = detect_layout(_trailing_sample(), list(TRAILING_JOINT_NAMES), "auto")
        assert layout == "trailing"

    def test_effector_alias_counts_as_gripper(self):
        names = list(INTERLEAVED_JOINT_NAMES)
        names[7] = "astribot_effector_left"
        assert detect_layout([0.0] * 16, names, "auto") == "interleaved"

    def test_qpos7_gripper_mm_is_interleaved(self):
        values = [0.1] * 16
        values[7] = 50.0
        assert detect_layout(values, None, "auto") == "interleaved"

    def test_qpos7_negative_radian_is_trailing(self):
        values = [0.1] * 16
        values[7] = -0.35
        assert detect_layout(values, None, "auto") == "trailing"

    def test_flag_true_overrides_names(self):
        assert detect_layout(_trailing_sample(), list(TRAILING_JOINT_NAMES), "true") == "interleaved"

    def test_flag_false_overrides_names(self):
        assert detect_layout(_interleaved_sample(), list(INTERLEAVED_JOINT_NAMES), "false") == "trailing"

    def test_ambiguous_auto_defaults_to_trailing(self):
        values = [0.2] * 16
        assert detect_layout(values, None, "auto") == "trailing"


class TestJpegRoundTrip:
    def test_decode_matches_shape_and_keeps_red_patch(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:200, 100:200] = (255, 0, 0)
        b64 = encode_jpeg_b64(image)
        decoded = decode_jpeg_b64(b64)
        assert decoded is not None
        assert decoded.shape == (480, 640, 3)
        patch = decoded[120:180, 120:180]
        assert patch[:, :, 0].mean() > 200
        assert patch[:, :, 1].mean() < 40
        assert patch[:, :, 2].mean() < 40

    def test_empty_payload_returns_none(self):
        assert decode_jpeg_b64("") is None

    def test_wire_images_map_to_cam_keys(self):
        image = np.full((480, 640, 3), 32, dtype=np.uint8)
        b64 = encode_jpeg_b64(image)
        message = {"images": dict.fromkeys(WIRE_IMAGE_KEYS, b64)}
        frames = decode_wire_images(message, 480, 640)
        assert set(frames) == set(CAMERA_KEYS)
        for frame in frames.values():
            assert frame.shape == (480, 640, 3)


class TestActionAndObsFeatures:
    def test_action_feature_order_matches_dataset_trailing_names(self):
        robot = Astribot(AstribotConfig())
        assert tuple(robot.action_features) == TRAILING_JOINT_NAMES
        assert all(ftype is float for ftype in robot.action_features.values())

    def test_observation_features_include_joints_then_cameras(self):
        robot = Astribot(AstribotConfig())
        keys = list(robot.observation_features)
        assert keys[:N_JOINTS] == list(TRAILING_JOINT_NAMES)
        assert keys[N_JOINTS:] == list(CAMERA_KEYS)
        for cam in CAMERA_KEYS:
            assert robot.observation_features[cam] == (480, 640, 3)

    def test_hw_features_become_dataset_cam_and_state_keys(self):
        robot = Astribot(AstribotConfig())
        feats = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
        assert feats["observation.state"]["names"] == list(TRAILING_JOINT_NAMES)
        assert feats["observation.state"]["shape"] == (16,)
        for cam in CAMERA_KEYS:
            key = f"observation.images.{cam}"
            assert key in feats
            assert feats[key]["shape"] == (480, 640, 3)

    def test_build_action_message_interleaved_clips_grippers(self):
        trailing = _trailing_sample()
        trailing[14] = 150.0
        trailing[15] = -5.0
        payload = build_action_message(trailing, "interleaved")
        assert payload["layout"] == "interleaved"
        assert payload["names"] == list(INTERLEAVED_JOINT_NAMES)
        assert payload["values"][7] == pytest.approx(100.0)  # left gripper on the wire
        assert payload["values"][15] == pytest.approx(0.0)
        recovered = interleaved_to_trailing(payload["values"])
        assert recovered[14] == pytest.approx(100.0)
        assert recovered[15] == pytest.approx(0.0)

    def test_extract_state_accepts_qpos_alias(self):
        values, names = extract_state({"qpos": _trailing_sample(), "names": list(TRAILING_JOINT_NAMES)})
        assert len(values) == 16
        assert names[7] == "arm_right_j0"

    def test_pack_unpack_length_prefixed_json(self):
        original = {"type": "obs", "state": {"values": [1.0] * 16, "names": ["j"] * 16}}
        assert unpack_message(pack_message(original)) == original


class TestAstribotConfig:
    def test_rejects_unknown_layout_flag(self):
        with pytest.raises(ValueError, match="robot_interleaved"):
            AstribotConfig(robot_interleaved="maybe")

    def test_normalizes_flag_case(self):
        cfg = AstribotConfig(robot_interleaved="TRUE")
        assert cfg.robot_interleaved == "true"
