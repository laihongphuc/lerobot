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
"""Configuration for the Astribot S1 TCP-bridge robot."""

from dataclasses import dataclass

from ..config import RobotConfig

_VALID_LAYOUT_FLAGS = ("auto", "true", "false")


@RobotConfig.register_subclass("astribot")
@dataclass
class AstribotConfig(RobotConfig):
    """Connect to the robot-PC SDK bridge over length-prefixed JSON/TCP.

    The policy and LeRobot dataset use **trailing** 16-D order
    ``[L arm 7, R arm 7, gripper_left, gripper_right]``. The robot wire is
    often **interleaved** ``[L arm 7, gripper_left, R arm 7, gripper_right]``.
    ``robot_interleaved`` controls detection:

    - ``auto``: use obs ``names`` if present, else ``qpos[7]`` units, else trailing
    - ``true``: always treat the wire as interleaved
    - ``false``: always treat the wire as trailing
    """

    host: str = "127.0.0.1"
    port: int = 5555
    robot_interleaved: str = "auto"
    image_height: int = 480
    image_width: int = 640
    connect_timeout_s: float = 10.0
    obs_timeout_s: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        flag = self.robot_interleaved.lower()
        if flag not in _VALID_LAYOUT_FLAGS:
            raise ValueError(
                f"robot_interleaved must be one of {_VALID_LAYOUT_FLAGS}, got {self.robot_interleaved!r}"
            )
        self.robot_interleaved = flag
