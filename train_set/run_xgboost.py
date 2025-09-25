#!/usr/bin/env python3
"""
Transformer/XGBoost Hardware Runner
Real-time inference using trained model (PyTorch .pth or XGBoost .joblib)
- If --model ends with .joblib -> use XGBoost (handles NaN natively)
- Else -> fallback to PyTorch (.pth)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import json
import argparse
import logging
from datetime import datetime
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import sys
import os
import subprocess
import platform
import threading
from queue import Queue, Empty

import joblib  # <-- XGBoost 모델 로딩용

# Jetson optimization imports
try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Add DynamixelSDK path
sys.path.append(os.path.join(os.path.dirname(__file__), 'DynamixelSDK', 'src', 'dynamixel_sdk'))

try:
    from dynamixel_sdk import *
except ImportError:
    print("⚠️ DynamixelSDK not available - running in test mode only")
    DynamixelSDK_available = False
else:
    DynamixelSDK_available = True

# -----------------------------
# Minimal PyTorch model (fallback for .pth)
# -----------------------------
class SimpleTransformer(nn.Module):
    """Ultra-simple neural network for regression"""

    def __init__(self, input_dim=4, output_dim=4, d_model=8, nhead=1,
                 num_layers=1, dim_feedforward=12, dropout=0.0):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

# -----------------------------
# Runner
# -----------------------------
class TransformerHardwareRunner:
    def __init__(self, model_path, hardware_config_path='hardware_config.json',
                 test_mode=False, target_fps=2000.0, show_display=False, use_tensorrt=True):
        self.model_path = model_path
        self.test_mode = test_mode
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Jetson optimization setup
        self.setup_jetson_optimization()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Display configuration
        self.show_display = show_display

        # Optimize CUDA memory before loading model (harmless if we use XGBoost)
        self.optimize_cuda_memory()

        # Load model (XGBoost .joblib OR PyTorch .pth)
        self.load_model()

        # Load hardware configuration
        self.load_hardware_config(hardware_config_path)

        # Initialize threaded camera capture first
        self.frame_queues = {}
        self.camera_threads = {}
        self.capture_active = False

        # Initialize cameras and MediaPipe
        self.camera_color_converters = {}
        self.setup_cameras()
        self.setup_mediapipe()
        if self.show_display:
            self.setup_display_windows()

        # Start threaded camera capture
        self.start_camera_threads()

        # Initialize servos (if not in test mode)
        if not test_mode and DynamixelSDK_available:
            self.setup_servos()
        else:
            self.logger.info("🧪 TEST MODE - Hardware control disabled")
            # Set default servo limits for test mode
            self.setup_servo_defaults()

        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.last_fps_time = time.time()

        # Safety and stability tracking
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_successful_positions = [2048, 3328, 1140, 3072]  # Safe default positions
        self.position_smoothing_alpha = 0.2  # Minimal smoothing for maximum response speed
        self.last_positions = [2048, 3328, 1140, 3072]

        # Safety limits
        self.emergency_stop = False
        self.safe_zone_min = [1280, 1920, 1120, 1664]
        self.safe_zone_max = [2944, 3456, 3200, 3136]

    # -----------------------------
    # System setup helpers
    # -----------------------------
    def setup_jetson_optimization(self):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'jetson' in model.lower():
                    self.is_jetson = True
                    self.logger.info(f"Detected Jetson device: {model}")
                    try:
                        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True, capture_output=True)
                        self.logger.info("Set Jetson to maximum performance mode")
                    except subprocess.CalledProcessError:
                        self.logger.warning("Could not set performance mode (sudo required)")
                    try:
                        subprocess.run(['sudo', 'jetson_clocks'], check=True, capture_output=True)
                        self.logger.info("Enabled maximum CPU/GPU clocks")
                    except subprocess.CalledProcessError:
                        self.logger.warning("Could not set maximum clocks (sudo required)")
                else:
                    self.is_jetson = False
        except FileNotFoundError:
            self.is_jetson = False

    def _optimize_with_tensorrt(self, model):
        try:
            self.logger.info("Optimizing model with TensorRT...")
            example_input = torch.randn(1, 4, device=self.device)
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,
                min_block_size=1,
                torch_executed_ops={"aten::linear"},
                optimization_level=5,
            )
            with torch.no_grad():
                _ = trt_model(example_input)
            self.logger.info("✅ TensorRT optimization successful")
            return trt_model
        except Exception as e:
            self.logger.warning(f"TensorRT optimization failed: {e}")
            self.logger.info("Continuing with standard PyTorch model")
            return model

    def optimize_cuda_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)
            except Exception:
                pass
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except Exception:
                pass
            self.logger.info("CUDA memory optimized")

    # -----------------------------
    # Model loading (XGB or Torch)
    # -----------------------------
    def load_model(self):
        """Load trained model (.joblib for XGBoost, .pth for PyTorch)."""
        self.model_type = None
        self.normalize = False
        self.scaler_X = None
        self.scaler_y = None

        # 1) XGBoost path
        if self.model_path.lower().endswith(".joblib"):
            self.logger.info(f"Loading XGBoost model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_type = "xgb"
            self.logger.info("XGBoost model loaded (NaN inputs supported).")
            return

        # 2) PyTorch path (fallback)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = checkpoint['model_config']

            self.model = SimpleTransformer(
                input_dim=4,
                output_dim=4,
                d_model=config.get('d_model', 8),
                nhead=config.get('nhead', 1),
                num_layers=config.get('num_layers', 1),
                dim_feedforward=config.get('dim_feedforward', 12),
                dropout=0.0
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Optional TensorRT
            if self.use_tensorrt and torch.cuda.is_available():
                self.model = self._optimize_with_tensorrt(self.model)

            # Load scalers if present
            self.scaler_X = checkpoint.get('scaler_X', None)
            self.scaler_y = checkpoint.get('scaler_y', None)
            self.normalize = checkpoint.get('normalize', False)

            self.model_type = "torch"
            self.logger.info(f"PyTorch model loaded from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def load_hardware_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Hardware config loaded: {config_path}")
        except FileNotFoundError:
            self.logger.warning(f"Hardware config not found: {config_path}, using defaults")
            self.config = {
                "servos": {
                    "port": "/dev/ttyUSB0",
                    "baudrate": 1000000,
                    "ids": [1, 2, 3, 4],
                    "min_positions": [1280, 1920, 1120, 1664],
                    "max_positions": [2944, 3456, 3200, 3136]
                },
                "cameras": {
                    "left": {"id": 0, "width": 640, "height": 480},
                    "right": {"id": 2, "width": 640, "height": 480}
                }
            }

    # -----------------------------
    # Cameras & display
    # -----------------------------
    def _camera_capture_thread(self, camera_name, camera):
        while self.capture_active:
            try:
                ret, frame = camera.read()
                if ret:
                    converter = self.camera_color_converters.get(camera_name)
                    if converter is not None:
                        try:
                            frame = cv2.cvtColor(frame, converter)
                        except cv2.error as error:
                            self.logger.warning(f"Color conversion failed for {camera_name} camera: {error}")
                            self.camera_color_converters[camera_name] = None
                    try:
                        self.frame_queues[camera_name].put_nowait(frame)
                    except:
                        try:
                            self.frame_queues[camera_name].get_nowait()
                            self.frame_queues[camera_name].put_nowait(frame)
                        except Empty:
                            pass
                else:
                    try:
                        self.frame_queues[camera_name].put_nowait(None)
                    except:
                        pass
            except Exception as e:
                self.logger.error(f"Camera {camera_name} capture error: {e}")
                time.sleep(0.001)

    def start_camera_threads(self):
        self.capture_active = True
        for camera_name, camera in self.cameras.items():
            if camera is not None:
                self.frame_queues[camera_name] = Queue(maxsize=2)
                thread = threading.Thread(
                    target=self._camera_capture_thread,
                    args=(camera_name, camera),
                    daemon=True
                )
                thread.start()
                self.camera_threads[camera_name] = thread
                self.logger.info(f"Started capture thread for {camera_name} camera")

    def stop_camera_threads(self):
        self.capture_active = False
        for camera_name, thread in self.camera_threads.items():
            thread.join(timeout=1.0)
            self.logger.info(f"Stopped capture thread for {camera_name} camera")
        self.camera_threads.clear()

    def get_latest_frames(self):
        frames = {}
        for camera_name in self.cameras.keys():
            try:
                frame = None
                while True:
                    try:
                        frame = self.frame_queues[camera_name].get_nowait()
                    except Empty:
                        break
                frames[camera_name] = frame
            except KeyError:
                frames[camera_name] = None
        return frames

    def setup_cameras(self):
        self.cameras = {}
        camera_config = self.config.get('cameras', {})

        left_config = camera_config.get('cam_left', {'id': 0, 'enabled': True})
        if left_config.get('enabled', True):
            left_id = left_config.get('id', 0)
            self.cameras['left'] = self._setup_single_camera('left', left_id, left_config)
        else:
            self.cameras['left'] = None
            self.logger.info("Left camera disabled in configuration")

        right_config = camera_config.get('cam_right', {'id': 2, 'enabled': True})
        if right_config.get('enabled', True):
            right_id = right_config.get('id', 2)
            self.cameras['right'] = self._setup_single_camera('right', right_id, right_config)
        else:
            self.cameras['right'] = None
            self.logger.info("Right camera disabled in configuration")

    def _setup_single_camera(self, camera_name, camera_id, config):
        camera = None
        if hasattr(self, 'is_jetson') and self.is_jetson and camera_id in [0, 1]:
            try:
                gst_pipeline = (
                    f"nvarguscamerasrc sensor-id={camera_id} ! "
                    f"video/x-raw(memory:NVMM), width=640, height=480, framerate=120/1 ! "
                    f"nvvidconv flip-method=0 ! "
                    f"video/x-raw, width=640, height=480, format=BGRx ! "
                    f"videoconvert ! "
                    f"video/x-raw, format=BGR ! "
                    f"appsink max-buffers=1 drop=true"
                )
                camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if camera.isOpened():
                    self.logger.info(f"{camera_name.title()} camera: CSI via GStreamer (sensor-id={camera_id})")
                    return camera
                else:
                    camera.release()
                    camera = None
            except Exception as e:
                self.logger.warning(f"GStreamer CSI failed for {camera_name}: {e}")

        try:
            camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            if not camera.isOpened():
                camera = cv2.VideoCapture(camera_id)

            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 120)
                fourcc = self._apply_camera_fourcc(camera_name, config.get('fourcc', 'MJPG'), camera)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_EXPOSURE, -6)
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                self.logger.info(f"{camera_name.title()} camera: USB via V4L2 (id={camera_id}, fourcc={fourcc or 'auto'})")
                return camera
            else:
                self.logger.warning(f"{camera_name.title()} camera not available (id={camera_id})")
                return None
        except Exception as e:
            self.logger.error(f"Failed to setup {camera_name} camera: {e}")
            return None

    def _apply_camera_fourcc(self, camera_name, preferred_fourcc, camera):
        if camera is None:
            self.camera_color_converters[camera_name] = None
            return None
        if preferred_fourcc:
            code = preferred_fourcc.upper()
            if len(code) == 4:
                try:
                    fourcc_value = cv2.VideoWriter_fourcc(*code)
                    if not camera.set(cv2.CAP_PROP_FOURCC, fourcc_value):
                        self.logger.warning(f"Failed to set FOURCC {code} for {camera_name} camera; device kept default")
                except Exception as error:
                    self.logger.warning(f"FOURCC {code} not supported for {camera_name} camera: {error}")
            else:
                self.logger.warning(f"Invalid FOURCC '{preferred_fourcc}' for {camera_name} camera; using device default")
        actual_code = self._read_fourcc(camera_name, camera)
        self.camera_color_converters[camera_name] = self._color_converter_for_fourcc(actual_code)
        return actual_code

    def _read_fourcc(self, camera_name, camera):
        try:
            value = int(camera.get(cv2.CAP_PROP_FOURCC))
        except Exception:
            return ''
        if value == 0:
            return ''
        chars = [chr((value >> (8 * i)) & 0xFF) for i in range(4)]
        return ''.join(chars).strip()

    def _color_converter_for_fourcc(self, fourcc_code):
        if not fourcc_code:
            return None
        code = fourcc_code.upper()
        if code in {'YUYV', 'YUY2', 'YUNV'}:
            return getattr(cv2, 'COLOR_YUV2BGR_YUY2', None)
        if code in {'UYVY', 'YVYU'}:
            return getattr(cv2, 'COLOR_YUV2BGR_UYVY', None)
        if code in {'BGR3', 'BGR4', 'BGRA', 'RGB3', 'RGB4', 'RGBA', 'MJPG'}:
            return None
        return None

    def setup_display_windows(self):
        if self.cameras.get('left') is not None:
            cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Left Camera', 640, 480)
            cv2.moveWindow('Left Camera', 100, 100)
            self.logger.info("Left camera window created")
        if self.cameras.get('right') is not None:
            cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Right Camera', 640, 480)
            cv2.moveWindow('Right Camera', 780, 100)
            self.logger.info("Right camera window created")
        self.logger.info("Display windows initialized - press 'q' to quit, 's' for emergency stop")

    def update_display(self, frames, left_features, right_features):
        if not self.show_display:
            return

        # Left
        left_frame = frames.get('left')
        if left_frame is not None:
            display_frame = left_frame.copy()
            x, y = left_features
            self._add_camera_overlay(display_frame, 'LEFT', x, y, left_features, right_features)
            # draw only if finite coords
            if np.isfinite([x, y]).all():
                cv2.circle(display_frame, (int(x), int(y)), 12, (0, 255, 0), -1)
                cv2.circle(display_frame, (int(x), int(y)), 15, (255, 255, 255), 2)
                coord_text = f"Index: ({int(x)}, {int(y)})"
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = int(x) + 20, int(y) - 20
                if text_x + text_size[0] > display_frame.shape[1]:
                    text_x = int(x) - text_size[0] - 20
                if text_y - text_size[1] < 0:
                    text_y = int(y) + 40
                cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, coord_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Left Camera", display_frame)

        # Right
        right_frame = frames.get('right')
        if right_frame is not None:
            display_frame = right_frame.copy()
            x, y = right_features
            self._add_camera_overlay(display_frame, 'RIGHT', x, y, left_features, right_features)
            if np.isfinite([x, y]).all():
                cv2.circle(display_frame, (int(x), int(y)), 12, (0, 255, 255), -1)
                cv2.circle(display_frame, (int(x), int(y)), 15, (255, 255, 255), 2)
                coord_text = f"Index: ({int(x)}, {int(y)})"
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = int(x) + 20, int(y) - 20
                if text_x + text_size[0] > display_frame.shape[1]:
                    text_x = int(x) - text_size[0] - 20
                if text_y - text_size[1] < 0:
                    text_y = int(y) + 40
                cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, coord_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Right Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord('s'):
            self.logger.warning('Emergency stop triggered via display window')
            self.emergency_stop = True

    def _add_camera_overlay(self, frame, camera_name, x, y, left_coords, right_coords):
        h, w = frame.shape[:2]
        # 'finite' check로 NaN 걸러냄
        hand_detected = np.isfinite([x, y]).all()
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if hand_detected else "NO HAND"

        overlay_height = 140
        cv2.rectangle(frame, (10, 10), (350, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, overlay_height), status_color, 2)

        cv2.putText(frame, f"{camera_name} CAMERA", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        if hand_detected:
            cv2.putText(frame, f"This: ({int(x)}, {int(y)})", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "This: (---, ---)", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

        left_x, left_y = left_coords
        right_x, right_y = right_coords
        left_valid = np.isfinite([left_x, left_y]).all()
        right_valid = np.isfinite([right_x, right_y]).all()
        left_text = f"L:({int(left_x)},{int(left_y)})" if left_valid else "L:(---,---)"
        right_text = f"R:({int(right_x)},{int(right_y)})" if right_valid else "R:(---,---)"
        cv2.putText(frame, f"Stereo: {left_text} | {right_text}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' for emergency stop",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # -----------------------------
    # MediaPipe (좌/우 별도) & features
    # -----------------------------
    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands_left = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2,
            model_complexity=0
        )
        self.hands_right = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def setup_servo_defaults(self):
        robot_config = self.config.get('robot_arms', {})
        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4])
        self.min_positions = [1024, 1024, 1024, 1024]
        self.max_positions = [2944, 3456, 3200, 3136]

    def setup_servos(self):
        if not DynamixelSDK_available:
            return
        robot_config = self.config.get('robot_arms', {})
        follower_config = robot_config.get('follower', {})
        port = follower_config.get('port', '/dev/follower_arm')
        baudrate = follower_config.get('baudrate', 1000000)
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(robot_config.get('protocol_version', 2.0))
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {port}")
        if not self.port_handler.setBaudRate(baudrate):
            raise Exception(f"Failed to set baudrate {baudrate}")
        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4])
        self.min_positions = [1280, 1920, 1120, 1664]
        self.max_positions = [3072, 3072, 3072, 3072]
        goal_position_addr = robot_config.get('addr_goal_position', 116)
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, goal_position_addr, 4)
        self.enable_torque()
        self.logger.info(f"Servos initialized on {port}")

    def enable_torque(self):
        if not DynamixelSDK_available:
            return
        torque_enable_addr = 64
        for servo_id in self.servo_ids:
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, torque_enable_addr, 1
                )
                if dxl_comm_result != COMM_SUCCESS:
                    self.logger.error(f"Failed to enable torque for servo {servo_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    self.logger.error(f"Servo {servo_id} error: {self.packet_handler.getRxPacketError(dxl_error)}")
                else:
                    self.logger.info(f"Torque enabled for servo {servo_id}")
            except Exception as e:
                self.logger.error(f"Exception enabling torque for servo {servo_id}: {e}")

    def disable_torque(self):
        if not DynamixelSDK_available:
            return
        torque_enable_addr = 64
        for servo_id in self.servo_ids:
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, torque_enable_addr, 0
                )
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    self.logger.info(f"Torque disabled for servo {servo_id}")
            except Exception as e:
                self.logger.error(f"Exception disabling torque for servo {servo_id}: {e}")

    def extract_hand_features(self, frame, camera_name):
        """Return [x,y] in pixels or [nan,nan] if not detected."""
        if frame is None:
            return [np.nan, np.nan], None

        try:
            if frame.size == 0 or len(frame.shape) != 3:
                self.logger.warning(f"Invalid frame from {camera_name}")
                return [np.nan, np.nan], frame.copy()

            hands_processor = self.hands_left if camera_name == 'left' else self.hands_right
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_processor.process(rgb_frame)
            frame.flags.writeable = True
            display_frame = frame.copy()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[8]
                if 0 <= index_tip.x <= 1 and 0 <= index_tip.y <= 1:
                    h, w = frame.shape[:2]
                    x = index_tip.x * w
                    y = index_tip.y * h
                    if 0 <= x <= w and 0 <= y <= h:
                        return [float(x), float(y)], display_frame

            # not detected
            return [np.nan, np.nan], display_frame

        except Exception as e:
            self.logger.error(f"Hand detection error in {camera_name}: {e}")
            return [np.nan, np.nan], frame.copy()

    # -----------------------------
    # Prediction (XGB or Torch)
    # -----------------------------
    def predict_joint_positions(self, features):
        """Predict joint positions using loaded model. Accepts NaNs if XGBoost."""
        try:
            features = np.array(features, dtype=np.float32).reshape(1, -1)

            if self.model_type == "xgb":
                # XGBoost는 NaN 허용 (default direction으로 분기)
                predictions = self.model.predict(features)
                result = predictions[0]
                if not np.isfinite(result).all():
                    self.logger.warning("Non-finite XGB predictions, using last successful positions")
                    return self.last_successful_positions.copy()
                # PyTorch 스케일러/역변환 없음
                return result

            # ----- PyTorch 경로 (기존 유지) -----
            # PyTorch 쪽은 NaN 입력을 처리하지 못하므로, NaN이 있으면 마지막 안전값을 유지
            if not np.isfinite(features).all():
                self.logger.warning("Non-finite features for Torch model, using defaults")
                return self.last_successful_positions.copy()

            if self.normalize and self.scaler_X is not None:
                features = self.scaler_X.transform(features)

            features_tensor = torch.FloatTensor(features).to(self.device)
            with torch.no_grad():
                predictions = self.model(features_tensor).cpu().numpy()

            if self.normalize and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions)

            result = predictions[0]
            if not np.isfinite(result).all():
                self.logger.warning("Non-finite Torch predictions, using last successful positions")
                return self.last_successful_positions.copy()
            return result

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.last_successful_positions.copy()

    # -----------------------------
    # Safety & control
    # -----------------------------
    def clamp_positions(self, positions):
        if self.emergency_stop:
            self.logger.warning("Emergency stop active - holding position")
            return self.last_positions.copy()

        safe_positions = []
        for i, pos in enumerate(positions):
            min_pos = self.safe_zone_min[i]
            max_pos = self.safe_zone_max[i]
            safe_pos = max(min_pos, min(max_pos, int(pos)))
            if i < len(self.last_positions):
                last_pos = self.last_positions[i]
                smoothed_pos = (self.position_smoothing_alpha * safe_pos +
                                (1 - self.position_smoothing_alpha) * last_pos)
                safe_pos = int(smoothed_pos)
            safe_positions.append(safe_pos)

        self.last_positions = safe_positions.copy()
        self.check_safety_violations(safe_positions)
        return safe_positions

    def check_safety_violations(self, positions):
        for i, pos in enumerate(positions):
            min_safe = self.safe_zone_min[i] if i < len(self.safe_zone_min) else 1200
            max_safe = self.safe_zone_max[i] if i < len(self.safe_zone_max) else 2896
            if pos < min_safe or pos > max_safe:
                self.logger.error(f"Safety violation: Joint {i} at position {pos}, safe range [{min_safe}-{max_safe}]")
                self.emergency_stop = True
                return
        if self.emergency_stop:
            all_safe = all(self.safe_zone_min[i] <= pos <= self.safe_zone_max[i]
                           for i, pos in enumerate(positions) if i < len(self.safe_zone_min))
            if all_safe:
                self.logger.info("Positions back in safe zone, resetting emergency stop")
                self.emergency_stop = False

    def send_servo_commands(self, positions):
        if self.test_mode or not DynamixelSDK_available:
            return True
        if self.emergency_stop:
            self.logger.warning("Emergency stop active - not sending commands")
            return False
        try:
            for i, pos in enumerate(positions):
                min_safe = self.safe_zone_min[i] if i < len(self.safe_zone_min) else 1200
                max_safe = self.safe_zone_max[i] if i < len(self.safe_zone_max) else 2896
                if pos < min_safe or pos > max_safe:
                    self.logger.error(f"Refusing to send unsafe position {pos} to joint {i}, safe range [{min_safe}-{max_safe}]")
                    self.emergency_stop = True
                    return False

            self.group_sync_write.clearParam()
            for i, servo_id in enumerate(self.servo_ids):
                if i < len(positions):
                    position = positions[i]
                    position_bytes = [
                        DXL_LOBYTE(DXL_LOWORD(position)),
                        DXL_HIBYTE(DXL_LOWORD(position)),
                        DXL_LOBYTE(DXL_HIWORD(position)),
                        DXL_HIBYTE(DXL_HIWORD(position))
                    ]
                    self.group_sync_write.addParam(servo_id, position_bytes)

            dxl_comm_result = self.group_sync_write.txPacket()
            success = dxl_comm_result == COMM_SUCCESS
            if success:
                self.last_successful_positions = positions.copy()
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.logger.warning(f"Servo command failed, consecutive failures: {self.consecutive_failures}")
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.logger.error("Too many consecutive failures, activating emergency stop")
                    self.emergency_stop = True
            return success

        except Exception as e:
            self.logger.error(f"Servo command failed: {e}")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.emergency_stop = True
            return False

    # -----------------------------
    # Control loop
    # -----------------------------
    def run_control_loop(self):
        self.logger.info(f"🧪 TEST Starting control loop at {self.target_fps:.2f} Hz" if self.test_mode
                         else f"🤖 Starting control loop at {self.target_fps:.2f} Hz")
        try:
            while True:
                loop_start = time.time()

                frames = self.get_latest_frames()

                # 좌/우에서 좌표 추출 (못 잡으면 NaN,NaN)
                left_features, left_processed_frame = self.extract_hand_features(frames.get('left'), 'left')
                right_features, right_processed_frame = self.extract_hand_features(frames.get('right'), 'right')

                if left_processed_frame is not None:
                    frames['left'] = left_processed_frame
                if right_processed_frame is not None:
                    frames['right'] = right_processed_frame

                if self.show_display:
                    self.update_display(frames, left_features, right_features)

                # [left_x, left_y, right_x, right_y]  — NaN 포함 가능
                combined_features = left_features + right_features

                inference_start = time.time()
                predicted_positions = self.predict_joint_positions(combined_features)
                inference_time = time.time() - inference_start

                clamped_positions = self.clamp_positions(predicted_positions)

                if not self.test_mode:
                    if not self.send_servo_commands(clamped_positions):
                        self.logger.warning("Failed to send servo commands, using safe position")
                        safe_positions = [2048, 3328, 1140, 3072]
                        self.send_servo_commands(safe_positions)

                self.frame_count += 1
                self.total_inference_time += inference_time

                if self.frame_count % 120 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.last_fps_time
                    fps = 120 / elapsed
                    avg_inference = (self.total_inference_time / self.frame_count) * 1000
                    mode_prefix = "🧪" if self.test_mode else "🤖"
                    self.logger.info(f"{mode_prefix} FPS: {fps:.1f}, Avg inference: {avg_inference:.1f}ms")
                    self.last_fps_time = current_time

                if self.test_mode and self.frame_count % 120 == 1:
                    self.logger.info(f"🧪 Features: {combined_features} → Predicted: {clamped_positions}")

        except KeyboardInterrupt:
            self.logger.info("Control loop stopped by user")
            self.stop_camera_threads()
            if not self.test_mode:
                safe_positions = [2048, 3328, 1140, 3072]
                self.send_servo_commands(safe_positions)
                self.logger.info("Returned to safe position")
        except Exception as e:
            self.logger.error(f"Control loop error: {e}")
            self.stop_camera_threads()
            if not self.test_mode:
                try:
                    safe_positions = [2048, 3328, 1140, 3072]
                    self.send_servo_commands(safe_positions)
                    self.logger.info("Emergency return to safe position")
                except:
                    self.logger.error("Failed to return to safe position")
            raise

    def cleanup(self):
        if not self.test_mode and DynamixelSDK_available:
            try:
                safe_positions = [2048, 3328, 1140, 3072]
                self.send_servo_commands(safe_positions)
                time.sleep(0.5)
                self.logger.info("Returned to safe position during cleanup")
            except Exception as e:
                self.logger.error(f"Failed to return to safe position: {e}")

        if not self.test_mode and DynamixelSDK_available:
            try:
                self.disable_torque()
            except Exception as e:
                self.logger.error(f"Failed to disable torque: {e}")

        for camera in self.cameras.values():
            if camera is not None:
                try:
                    camera.release()
                except Exception as e:
                    self.logger.error(f"Failed to release camera: {e}")

        if self.show_display:
            cv2.destroyAllWindows()

        if hasattr(self, 'port_handler') and not self.test_mode:
            try:
                self.port_handler.closePort()
            except Exception as e:
                self.logger.error(f"Failed to close port: {e}")

        self.logger.info("Cleanup completed")

# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Transformer/XGBoost Hardware Runner")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib for XGBoost, .pth for Torch)")
    parser.add_argument("--config", default="hardware_config.json", help="Hardware configuration file")
    parser.add_argument("--test", action='store_true', help="Run in test mode (no hardware control)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS (default: 60.0)")
    parser.add_argument("--display", action='store_true', help="Show OpenCV preview windows")
    parser.add_argument("--no-tensorrt", action='store_true', help="Disable TensorRT optimization (Torch only)")
    args = parser.parse_args()

    print("🤖 Hardware Runner (XGBoost/Torch)")
    print("=" * 50)

    try:
        runner = TransformerHardwareRunner(
            model_path=args.model,
            hardware_config_path=args.config,
            test_mode=args.test,
            target_fps=args.fps,
            show_display=args.display,
            use_tensorrt=not args.no_tensorrt
        )
        runner.run_control_loop()
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'runner' in locals():
            runner.cleanup()

if __name__ == "__main__":
    main()

#python run_xgboost.py --model xgb_cam2joint_overfit_20250923_221041.joblib --test
