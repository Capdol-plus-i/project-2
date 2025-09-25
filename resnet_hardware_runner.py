#!/usr/bin/env python3
"""
Real-time hardware runner that maps fingertip coordinates to follower arm poses
using the trained residual MLP (model_parameters_resnet.npz).

The script:
  * Streams frames from dual cameras and extracts index fingertip pixels via MediaPipe
  * Normalizes the 4-D feature vector and performs a forward pass through the saved model
  * Sends batched goal positions to the follower arm via the Dynamixel SDK GroupSyncWrite

Usage (activate your venv first if needed):
    python resnet_hardware_runner.py --model model_parameters_resnet.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import joblib
from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

# =======================
# === JSON UTILITIES ===
# =======================


def load_json(path: Path) -> Dict:
    """Load JSON configuration with a helpful error message."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


# =======================
# === MODEL INFERENCE ===
# =======================


class UniversalRegressor:
    """Universal model loader for both NPZ and PKL formats."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model_type = self._detect_model_type(model_path)

        if self.model_type == 'npz':
            self._load_npz_model(model_path)
        elif self.model_type == 'pkl':
            self._load_pkl_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        logger.info("Model loaded from %s (type: %s)", model_path, self.model_type)

    @staticmethod
    def _detect_model_type(model_path: Path) -> str:
        """Detect model type based on file extension."""
        suffix = model_path.suffix.lower()
        if suffix == '.npz':
            return 'npz'
        elif suffix == '.pkl':
            return 'pkl'
        else:
            raise ValueError(f"Unsupported model file extension: {suffix}")

    def _load_npz_model(self, npz_path: Path) -> None:
        """Load NPZ format model (train_song.py output)."""
        self.input_scale = 650.0
        self.output_scale = 4100.0

        data = np.load(str(npz_path))
        self.parameters = {name: data[name] for name in data.files}

        # Count only main weight layers (W1, W2, ..., W5), not residual weights (Wr1, Wr2)
        main_weight_keys = [k for k in self.parameters.keys() if k.startswith('W') and not k.startswith('Wr')]
        self.layer_count = len(main_weight_keys)

        # For NPZ models, we need to implement forward pass manually
        self.scaler_X = None
        self.scaler_y = None
        self.sklearn_model = None

    def _load_pkl_model(self, pkl_path: Path) -> None:
        """Load PKL format model (train_regression_model.py output)."""
        model_data = joblib.load(str(pkl_path))

        self.sklearn_model = model_data['model']
        self.scaler_X = model_data.get('scaler_X')
        self.scaler_y = model_data.get('scaler_y')
        self.normalize = model_data.get('normalize', True)
        self.feature_cols = model_data.get('feature_cols', ['cam_left_x', 'cam_left_y', 'cam_right_x', 'cam_right_y'])
        self.target_cols = model_data.get('target_cols', ['follower_pos1', 'follower_pos2', 'follower_pos3', 'follower_pos4'])

        # For PKL models, we don't need manual scaling
        self.parameters = None
        self.layer_count = None

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, alpha * z)

    def _predict_npz(self, features: Sequence[float]) -> np.ndarray:
        """Forward pass for NPZ models with residual connections."""
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (4,):
            raise ValueError(f"Expected feature vector of shape (4,), got {x.shape}")

        # Input normalization
        a = (x / self.input_scale).reshape(-1, 1)

        # Check if this is a residual model by looking for Wr1, Wr2
        is_residual = 'Wr1' in self.parameters and 'Wr2' in self.parameters

        if is_residual:
            # Residual network: 4 -> 64 -> 128 -> 64 -> 32 -> 4
            # Layer 1: 4 -> 64
            z1 = self.parameters['W1'] @ a + self.parameters['b1']
            a1 = self._leaky_relu(z1)  # 64 dim

            # Layer 2: 64 -> 128
            z2 = self.parameters['W2'] @ a1 + self.parameters['b2']
            a2 = self._leaky_relu(z2)  # 128 dim

            # Layer 3: 128 -> 64 with residual from a1
            z3 = self.parameters['W3'] @ a2 + self.parameters['b3']
            skip1 = self.parameters['Wr1'] @ a1  # 64 -> 64 projection
            a3 = self._leaky_relu(z3 + skip1)  # 64 dim with residual

            # Layer 4: 64 -> 32 with residual from a1
            z4 = self.parameters['W4'] @ a3 + self.parameters['b4']
            skip2 = self.parameters['Wr2'] @ a1  # 64 -> 32 projection
            a4 = self._leaky_relu(z4 + skip2)  # 32 dim with residual

            # Output layer: 32 -> 4
            output = self.parameters['W5'] @ a4 + self.parameters['b5']

        else:
            # Simple sequential network
            a_current = a
            for layer in range(1, self.layer_count):
                w = self.parameters[f"W{layer}"]
                b = self.parameters[f"b{layer}"]
                a_current = self._relu(w @ a_current + b)

            w_final = self.parameters[f"W{self.layer_count}"]
            b_final = self.parameters[f"b{self.layer_count}"]
            output = w_final @ a_current + b_final

        return (output.flatten() * self.output_scale).astype(np.float64)

    def _predict_pkl_residual(self, features: Sequence[float]) -> np.ndarray:
        """Forward pass for PKL residual models with manual implementation."""
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)

        if self.normalize and self.scaler_X is not None:
            x = self.scaler_X.transform(x)

        # Manual forward pass for residual MLP
        # Network: 4 -> 64 -> 128 -> 64 -> 32 -> 4 with residual connections

        # Layer 1: 4 -> 64
        z1 = np.dot(x, self.sklearn_model.weights['W1']) + self.sklearn_model.biases['b1']
        a1 = self._leaky_relu(z1)  # 64 dim

        # Layer 2: 64 -> 128
        z2 = np.dot(a1, self.sklearn_model.weights['W2']) + self.sklearn_model.biases['b2']
        a2 = self._leaky_relu(z2)  # 128 dim

        # Layer 3: 128 -> 64 with residual from a1
        z3 = np.dot(a2, self.sklearn_model.weights['W3']) + self.sklearn_model.biases['b3']
        skip1 = np.dot(a1, self.sklearn_model.weights['Wr1'])  # 64 -> 64 projection
        a3 = self._leaky_relu(z3 + skip1)  # 64 dim with residual

        # Layer 4: 64 -> 32 with residual from a1
        z4 = np.dot(a3, self.sklearn_model.weights['W4']) + self.sklearn_model.biases['b4']
        skip2 = np.dot(a1, self.sklearn_model.weights['Wr2'])  # 64 -> 32 projection
        a4 = self._leaky_relu(z4 + skip2)  # 32 dim with residual

        # Output layer: 32 -> 4
        z5 = np.dot(a4, self.sklearn_model.weights['W5']) + self.sklearn_model.biases['b5']
        output = z5  # No activation on output layer

        if self.normalize and self.scaler_y is not None:
            output = self.scaler_y.inverse_transform(output)

        return output.flatten().astype(np.float64)

    def predict(self, features: Sequence[float]) -> np.ndarray:
        """Universal prediction method."""
        if self.model_type == 'npz':
            return self._predict_npz(features)
        elif self.model_type == 'pkl':
            return self._predict_pkl_residual(features)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


# ==============================
# === CAMERA / HAND TRACKING ===
# ==============================


@dataclass
class CameraConfig:
    identifier: Optional[int]
    enabled: bool


class DualCameraHandTracker:
    """Streams from two cameras and extracts index-fingertip pixel coordinates."""

    def __init__(self, cam_left: CameraConfig, cam_right: CameraConfig, width: int = 640, height: int = 480, fps: int = 30,
                 ema_alpha: float = 0.35, show_display: bool = False) -> None:
        self.cam_left_cfg = cam_left
        self.cam_right_cfg = cam_right
        self.width = width
        self.height = height
        self.fps = fps
        self.ema_alpha = ema_alpha
        self.show_display = show_display

        self._mp = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils if show_display else None
        self._hands_left = self._mp.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.4)
        self._hands_right = self._mp.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.4)

        self._cap_left: Optional[cv2.VideoCapture] = None
        self._cap_right: Optional[cv2.VideoCapture] = None
        self._last_valid: Optional[np.ndarray] = None

        # Display variables
        self._display_frames = {'left': None, 'right': None}
        self._current_features = [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _resolve_identifier(identifier: Optional[int]) -> Optional[int]:
        if identifier is None:
            return None
        if isinstance(identifier, str) and identifier.startswith("/dev/") and "video" in identifier:
            try:
                return int(identifier.split("video")[-1])
            except ValueError:
                return None
        try:
            return int(identifier)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> None:
        """Open available cameras."""
        if self.cam_left_cfg.enabled:
            left_id = self._resolve_identifier(self.cam_left_cfg.identifier)
            self._cap_left = cv2.VideoCapture(left_id if left_id is not None else 0, cv2.CAP_V4L2)
            if self._cap_left.isOpened():
                self._configure_capture(self._cap_left, self.width, self.height, self.fps)
                logger.info("Left camera ready (id=%s)", left_id)
            else:
                logger.warning("Left camera not available (id=%s)", left_id)
                self._cap_left.release()
                self._cap_left = None

        if self.cam_right_cfg.enabled:
            right_id = self._resolve_identifier(self.cam_right_cfg.identifier)
            self._cap_right = cv2.VideoCapture(right_id if right_id is not None else 2, cv2.CAP_V4L2)
            if self._cap_right.isOpened():
                self._configure_capture(self._cap_right, self.width, self.height, self.fps)
                logger.info("Right camera ready (id=%s)", right_id)
            else:
                logger.warning("Right camera not available (id=%s)", right_id)
                self._cap_right.release()
                self._cap_right = None

        if not self._cap_left and not self._cap_right:
            raise RuntimeError("No camera could be initialized. Check hardware connections and IDs in hardware_config.json")

        # Setup display windows if requested
        if self.show_display:
            self._setup_display_windows()

    def _setup_display_windows(self) -> None:
        """Setup OpenCV display windows"""
        cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Left Camera', 640, 480)
        cv2.resizeWindow('Right Camera', 640, 480)

        # Position windows side by side
        cv2.moveWindow('Left Camera', 100, 100)
        cv2.moveWindow('Right Camera', 750, 100)

        logger.info("Display windows initialized - Press 'q' to quit")

    def _process_frame(self, cap: cv2.VideoCapture, hand_model: mp.solutions.hands.Hands, camera_name: str) -> Optional[Tuple[int, int]]:
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok:
            return None

        # Make a copy for display
        display_frame = frame.copy() if self.show_display else None

        frame.flags.writeable = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_model.process(rgb)
        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks on display frame if showing display
            if self.show_display and self._mp_drawing:
                self._mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, self._mp.HAND_CONNECTIONS)

            idx_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(idx_tip.x * w), int(idx_tip.y * h)

            # Draw index finger tip circle
            if self.show_display:
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Index: ({x}, {y})",
                          (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store display frame
            if self.show_display:
                self._display_frames[camera_name] = display_frame

            return x, y

        # Store display frame even if no hand detected
        if self.show_display:
            self._display_frames[camera_name] = display_frame

        return None

    def get_feature_vector(self) -> Optional[np.ndarray]:
        """Return [left_x, left_y, right_x, right_y] with EMA smoothing."""
        coords = np.full(4, np.nan, dtype=np.float64)
        left_tip = self._process_frame(self._cap_left, self._hands_left, 'left') if self._cap_left else None
        right_tip = self._process_frame(self._cap_right, self._hands_right, 'right') if self._cap_right else None

        if left_tip:
            coords[0], coords[1] = left_tip
        if right_tip:
            coords[2], coords[3] = right_tip

        if np.isnan(coords).all():
            return None if self._last_valid is None else self._last_valid.copy()

        if self._last_valid is None:
            self._last_valid = np.nan_to_num(coords, nan=0.0)
        else:
            current = np.nan_to_num(coords, nan=self._last_valid)
            self._last_valid = self.ema_alpha * current + (1.0 - self.ema_alpha) * self._last_valid

        # Store current features for display
        self._current_features = self._last_valid.copy().tolist()

        # Update display if requested
        if self.show_display:
            self._update_display()

        return self._last_valid.copy()

    def _update_display(self) -> None:
        """Update OpenCV display windows with overlay information"""
        try:
            # Add overlay information to frames
            for camera_name in ['left', 'right']:
                frame = self._display_frames.get(camera_name)
                if frame is not None:
                    # Add status overlay
                    self._add_overlay_info(frame, camera_name)

                    # Display frame
                    window_name = f"{camera_name.title()} Camera"
                    cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested via keyboard")
                import sys
                sys.exit(0)

        except Exception as e:
            logger.error(f"Display update error: {e}")

    def _add_overlay_info(self, frame, camera_name: str) -> None:
        """Add overlay information to the frame"""
        try:
            h, w = frame.shape[:2]

            # Add background rectangle for text
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 120), (0, 255, 0), 2)

            # Camera name and status
            cv2.putText(frame, f"{camera_name.upper()} CAMERA", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Status: RUNNING", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Current features (hand position)
            if len(self._current_features) >= 4:
                if camera_name == 'left':
                    x, y = self._current_features[0], self._current_features[1]
                else:
                    x, y = self._current_features[2], self._current_features[3]

                cv2.putText(frame, f"Hand: ({int(x)}, {int(y)})", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Control instructions
            cv2.putText(frame, "Press 'q' to quit",
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        except Exception as e:
            logger.error(f"Overlay error: {e}")

    def stop(self) -> None:
        if self._cap_left:
            self._cap_left.release()
            self._cap_left = None
        if self._cap_right:
            self._cap_right.release()
            self._cap_right = None
        self._hands_left.close()
        self._hands_right.close()

        # Close display windows
        if self.show_display:
            cv2.destroyAllWindows()


# ===================================
# === DYNAMIXEL HARDWARE INTERFACE ===
# ===================================


# DXL utility functions now imported from dynamixel_sdk


class DynamixelFollowerInterface:
    """Low-latency Dynamixel controller optimized for follower arm commands."""

    def __init__(self, robot_cfg: Dict, protocol_version: float = 2.0,
                 max_delta: float = 80.0) -> None:
        self.protocol_version = protocol_version
        self.addr_goal_position = int(robot_cfg.get("addr_goal_position", 116))
        self.addr_torque_enable = int(robot_cfg.get("addr_torque_enable", 64))
        self.addr_present_position = int(robot_cfg.get("addr_present_position", 132))
        self.motor_ids: List[int] = robot_cfg.get("motor_ids", [1, 2, 3, 4])

        follower_cfg = robot_cfg.get("follower", {})
        self.port_name = follower_cfg.get("port", "/dev/ttyACM0")
        self.baudrate = int(follower_cfg.get("baudrate", 1000000))
        self.enabled = follower_cfg.get("enabled", True)

        self.max_delta = abs(float(max_delta))
        self.port_handler: Optional[PortHandler] = None
        self.packet_handler: Optional[PacketHandler] = None
        self.sync_writer: Optional[GroupSyncWrite] = None
        self.sync_reader: Optional[GroupSyncRead] = None
        self._last_command: Optional[np.ndarray] = None
        self._last_feedback: Optional[np.ndarray] = None
        self._sync_prepared = False

    def connect(self) -> None:
        if not self.enabled:
            raise RuntimeError("Follower arm is disabled in hardware_config.json")

        # Check if robot_arms is enabled
        # This should be checked in main() but add defensive check here

        self.port_handler = PortHandler(self.port_name)
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port_name}")

        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.port_name}")

        self.packet_handler = PacketHandler(self.protocol_version)
        self.sync_writer = GroupSyncWrite(self.port_handler, self.packet_handler, self.addr_goal_position, 4)
        self.sync_reader = GroupSyncRead(self.port_handler, self.packet_handler, self.addr_present_position, 4)

        for motor_id in self.motor_ids:
            if self.sync_reader and not self.sync_reader.addParam(motor_id):
                logger.debug("GroupSyncRead param already set for motor %d", motor_id)

        for motor_id in self.motor_ids:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, self.addr_torque_enable, 1
            )
            if result != COMM_SUCCESS or error != 0:
                logger.warning("Torque enable failed for motor %d (result=%s, error=%s)", motor_id, result, error)

        # GroupSyncWrite doesn't need pre-initialization in newer SDK
        # Parameters are added dynamically during send_goal_positions()
        self._sync_prepared = True

        logger.info("Follower arm connected on %s @ %d bps", self.port_name, self.baudrate)

    def disconnect(self) -> None:
        if not self.port_handler or not self.packet_handler:
            return
        for motor_id in self.motor_ids:
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, self.addr_torque_enable, 0)
        self.port_handler.closePort()
        self.sync_reader = None
        self.sync_writer = None
        self._sync_prepared = False
        logger.info("Follower arm disconnected")

    def send_goal_positions(self, positions: Sequence[float]) -> None:
        if not self.sync_writer or not self._sync_prepared:
            raise RuntimeError("Dynamixel interface not connected or not initialized")
        target = np.asarray(positions, dtype=np.float64)

        feedback = None
        if self.sync_reader:
            result = self.sync_reader.fastSyncRead()
            if result != COMM_SUCCESS:
                result = self.sync_reader.txRxPacket()
            if result == COMM_SUCCESS:
                samples = []
                for motor_id in self.motor_ids:
                    if self.sync_reader.isAvailable(motor_id, self.addr_present_position, 4):
                        samples.append(self.sync_reader.getData(motor_id, self.addr_present_position, 4))
                    else:
                        samples = []
                        break
                if samples:
                    feedback = np.asarray(samples, dtype=np.float64)
                    self._last_feedback = feedback
        if feedback is None:
            feedback = self._last_feedback if self._last_feedback is not None else self._last_command

        baseline = feedback if feedback is not None else target
        delta = np.clip(target - baseline, -self.max_delta, self.max_delta)
        safe_target = np.clip(baseline + delta, 0.0, 4095.0)
        self._last_command = safe_target

        # Clear previous parameters
        self.sync_writer.clearParam()

        # Add new parameters for all motors
        for motor_id, position in zip(self.motor_ids, safe_target):
            value = int(round(position))
            value = max(0, min(4095, value))
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(value)),
                DXL_HIBYTE(DXL_LOWORD(value)),
                DXL_LOBYTE(DXL_HIWORD(value)),
                DXL_HIBYTE(DXL_HIWORD(value)),
            ]
            if not self.sync_writer.addParam(motor_id, param_goal_position):
                logger.warning("Failed to add param for motor %d", motor_id)
        result = self.sync_writer.txPacket()
        if result != COMM_SUCCESS:
            logger.warning("GroupSyncWrite failed (result=%s)", result)

# =========================
# === APPLICATION ENTRY ===
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trained neural network model on follower arm in real-time")
    parser.add_argument("--model", type=Path, default=Path("model_parameters_resnet.npz"),
                        help="Path to model file (.npz or .pkl format)")
    parser.add_argument("--hardware-config", type=Path, default=Path("hardware_config.json"),
                        help="Hardware configuration JSON")
    parser.add_argument("--rate", type=float, default=60.0, help="Control loop frequency (Hz)")
    parser.add_argument("--max-delta", type=float, default=80.0,
                        help="Maximum change per cycle in Dynamixel ticks (safety clamp)")
    parser.add_argument("--test", action='store_true',
                        help="Test mode - show predictions without controlling hardware")
    parser.add_argument("--display", action='store_true',
                        help="Show OpenCV camera windows with real-time visualization")
    return parser


def graceful_shutdown(hand_tracker: DualCameraHandTracker, follower: Optional[DynamixelFollowerInterface]) -> None:
    logger.info("Shutting down...")
    if follower:
        follower.disconnect()
    hand_tracker.stop()


def main(args: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    cli_args = parser.parse_args(args)

    config_payload = load_json(cli_args.hardware_config)

    # Check if robot_arms is enabled
    robot_arms_cfg = config_payload.get("robot_arms", {})
    if not robot_arms_cfg.get("enabled", True):
        logger.error("Robot arms are disabled in hardware_config.json")
        return 1

    hardware_cfg = robot_arms_cfg
    camera_cfg = config_payload.get("cameras", {})

    # Check if cameras are enabled
    if not camera_cfg.get("enabled", True):
        logger.error("Cameras are disabled in hardware_config.json")
        return 1

    left_cam_cfg = CameraConfig(
        identifier=camera_cfg.get("cam_left", {}).get("id", 0),
        enabled=bool(camera_cfg.get("cam_left", {}).get("enabled", True)),
    )
    right_cam_cfg = CameraConfig(
        identifier=camera_cfg.get("cam_right", {}).get("id", 2),
        enabled=bool(camera_cfg.get("cam_right", {}).get("enabled", True)),
    )

    hand_tracker = DualCameraHandTracker(left_cam_cfg, right_cam_cfg, show_display=cli_args.display)
    model = UniversalRegressor(cli_args.model)

    hand_tracker.start()

    follower = None
    if not cli_args.test:
        follower = DynamixelFollowerInterface(hardware_cfg, max_delta=cli_args.max_delta)
        follower.connect()
    else:
        logger.info("🧪 TEST MODE - Hardware control disabled")

    stop_flag = False

    def _signal_handler(_sig: int, _frame) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    loop_period = 1.0 / max(cli_args.rate, 1.0)
    mode_text = "🧪 TEST" if cli_args.test else "🚀 LIVE"
    logger.info("%s Starting control loop at %.2f Hz", mode_text, 1.0 / loop_period)

    frame_count = 0
    try:
        while not stop_flag:
            features = hand_tracker.get_feature_vector()
            if features is None:
                time.sleep(loop_period)
                continue

            predicted = model.predict(features)

            if cli_args.test:
                # Test mode: just log predictions
                if frame_count % 30 == 0:  # Log every 30 frames (~0.5s at 60Hz)
                    logger.info("🧪 Features: [%.1f, %.1f, %.1f, %.1f] → Predicted: [%.0f, %.0f, %.0f, %.0f]",
                              features[0], features[1], features[2], features[3],
                              predicted[0], predicted[1], predicted[2], predicted[3])
            else:
                # Live mode: control hardware
                follower.send_goal_positions(predicted)

            frame_count += 1
            time.sleep(loop_period)
    finally:
        graceful_shutdown(hand_tracker, follower)

    return 0


if __name__ == "__main__":
    sys.exit(main())
