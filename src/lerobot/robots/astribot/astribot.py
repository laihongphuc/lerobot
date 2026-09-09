#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or granted to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Astribot S1 client: TCP JSON bridge on the robot PC, trailing features for the policy."""

from __future__ import annotations

import json
import logging
import socket
import threading
from contextlib import suppress
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_astribot import AstribotConfig
from .protocol import (
    CAMERA_KEYS,
    TRAILING_JOINT_NAMES,
    Layout,
    build_action_message,
    decode_wire_images,
    detect_layout,
    extract_state,
    interleaved_to_trailing,
    recv_message,
    send_message,
)

logger = logging.getLogger(__name__)


class Astribot(Robot):
    """Talks to the Astribot SDK TCP bridge; exposes dataset-order joints and cameras.

    A background reader conflates incoming obs so ``get_observation`` always
    returns the latest frame. ``send_action`` writes one JSON action; the bridge
    is expected to apply it at 30 Hz and hold the last command on disconnect.
    This class never sends a zero command on teardown.
    """

    config_class = AstribotConfig
    name = "astribot"

    def __init__(self, config: AstribotConfig):
        super().__init__(config)
        self.config = config
        self.cameras: dict[str, Any] = {}
        self._socket: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._obs_lock = threading.Lock()
        self._latest_obs: dict[str, Any] | None = None
        self._obs_event = threading.Event()
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._is_connected = False
        self._layout: Layout | None = None
        self._last_images: dict[str, np.ndarray] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = dict.fromkeys(TRAILING_JOINT_NAMES, float)
        shape = (self.config.image_height, self.config.image_width, 3)
        for cam in CAMERA_KEYS:
            features[cam] = shape
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(TRAILING_JOINT_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("Calibration not required for Astribot TCP bridge")

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._socket = socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.connect_timeout_s,
            )
        except OSError as e:
            raise DeviceNotConnectedError(
                f"Cannot connect to Astribot bridge at {self.config.host}:{self.config.port}: {e}"
            ) from e
        self._socket.settimeout(1.0)
        self._stop_event.clear()
        self._obs_event.clear()
        self._latest_obs = None
        self._layout = None
        self._is_connected = True
        self._reader_thread = threading.Thread(target=self._read_loop, name="astribot-obs", daemon=True)
        self._reader_thread.start()
        if calibrate:
            self.calibrate()
        logger.info(
            "%s connected to TCP bridge %s:%s",
            self.name,
            self.config.host,
            self.config.port,
        )

    def _read_loop(self) -> None:
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                message = recv_message(self._socket)
            except TimeoutError:
                continue
            except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as e:
                if not self._stop_event.is_set():
                    logger.warning("Astribot TCP lost: %s", e)
                    self._is_connected = False
                break
            if not isinstance(message, dict):
                continue
            if message.get("type", "obs") != "obs":
                continue
            with self._obs_lock:
                self._latest_obs = message
            self._obs_event.set()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if not self._obs_event.wait(timeout=self.config.obs_timeout_s):
            raise DeviceNotConnectedError(
                f"Timed out after {self.config.obs_timeout_s}s waiting for an obs from the Astribot bridge"
            )
        with self._obs_lock:
            message = self._latest_obs
        if message is None:
            raise DeviceNotConnectedError("No observation received from the Astribot bridge")

        values, names = extract_state(message)
        layout = detect_layout(values, names, self.config.robot_interleaved)
        if self._layout is None:
            self._layout = layout
            logger.info(
                "Robot names say %s — matching reply layout",
                "interleaved [L7,gL,R7,gR]" if layout == "interleaved" else "trailing [L7,R7,gL,gR]",
            )
        elif layout != self._layout:
            logger.warning("Layout changed %s → %s; following the new obs", self._layout, layout)
            self._layout = layout

        trailing = interleaved_to_trailing(values) if self._layout == "interleaved" else list(values)
        frames = decode_wire_images(
            message,
            self.config.image_height,
            self.config.image_width,
            fallback=self._last_images,
        )
        self._last_images = frames

        observation: RobotObservation = {}
        for name, value in zip(TRAILING_JOINT_NAMES, trailing, strict=True):
            observation[name] = value
        observation.update(frames)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        trailing = [float(action[name]) for name in TRAILING_JOINT_NAMES]
        layout: Layout = self._layout or detect_layout(None, None, self.config.robot_interleaved)
        payload = build_action_message(trailing, layout)
        assert self._socket is not None
        try:
            with self._send_lock:
                send_message(self._socket, payload)
        except OSError as e:
            logger.error("Failed to send action to Astribot bridge: %s", e)
            self._is_connected = False
            raise DeviceNotConnectedError(f"Astribot TCP send failed: {e}") from e

        # Return dataset-order (trailing) values actually commanded after gripper clip.
        return {
            name: trailing[i] if i < 14 else min(100.0, max(0.0, trailing[i]))
            for i, name in enumerate(TRAILING_JOINT_NAMES)
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        # Do not send zeros — the bridge holds the last command.
        self._stop_event.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                sock.close()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        self._is_connected = False
        logger.info("%s disconnected (bridge should hold last command)", self.name)
