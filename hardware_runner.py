#!/usr/bin/env python3
"""
Transformer Hardware Runner
Real-time inference using trained transformer model for camera to joint mapping
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import json
import argparse
import logging
from typing import Sequence
from datetime import datetime
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import sys
import os
import subprocess
import platform
import threading
from queue import Queue, Empty

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

# Model classes for different architectures
class SimpleTransformer(nn.Module):
    """Ultra-simple neural network for regression"""

    def __init__(self, input_dim=4, output_dim=4, d_model=8, nhead=1,
                 num_layers=1, dim_feedforward=12, dropout=0.0):
        super().__init__()

        # Minimal 2-layer network
        self.network = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class ConfigurableFeedforward(nn.Module):
    """Feedforward network that mirrors training-time architecture."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        hidden_sizes: Sequence[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_tuple = tuple(hidden_sizes) if hidden_sizes is not None else ()
        if not hidden_tuple:
            raise ValueError("hidden_sizes must contain at least one layer")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_tuple:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with feedforward layers"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.proj(x)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = out + residual
        out = self.norm(out)
        return out


class ResFeedforward(nn.Module):
    """Residual feedforward network"""
    def __init__(self, input_dim=4, output_dim=4, dropout=0.0):
        super().__init__()
        self.block_a = ResidualBlock(input_dim, 8, 16, dropout)
        self.block_b = ResidualBlock(16, 8, output_dim, dropout)
        self.long_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = self.long_skip(x)
        out = self.block_a(x)
        out = self.block_b(out)
        out = out + residual
        return out

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

        # Device setup with CUDA optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            # Set CUDA optimization flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Display configuration
        self.show_display = show_display

        # Optimize CUDA memory before loading model
        self.optimize_cuda_memory()

        # Load model
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

        # Safety limits - no per-iteration rate limits
        self.emergency_stop = False
        self.safe_zone_min = [1280, 1920, 1120, 1664]  # Conservative safe limits
        self.safe_zone_max = [2944, 3456, 3200, 3136]

    def setup_jetson_optimization(self):
        """Setup Jetson-specific optimizations"""
        try:
            # Check if running on Jetson
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'jetson' in model.lower():
                    self.is_jetson = True
                    self.logger.info(f"Detected Jetson device: {model}")

                    # Set maximum performance mode
                    try:
                        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True, capture_output=True)
                        self.logger.info("Set Jetson to maximum performance mode")
                    except subprocess.CalledProcessError:
                        self.logger.warning("Could not set performance mode (sudo required)")

                    # Set maximum CPU frequency
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
        """Optimize model with TensorRT for maximum inference speed"""
        try:
            self.logger.info("Optimizing model with TensorRT...")

            # Create example input tensor
            example_input = torch.randn(1, 4, device=self.device)

            # TensorRT compilation settings for maximum speed
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16},  # Use FP16 for speed
                workspace_size=1 << 30,  # 1GB workspace
                min_block_size=1,
                torch_executed_ops={"aten::linear"},  # Keep some ops in PyTorch
                optimization_level=5,  # Maximum optimization
            )

            # Test the optimized model
            with torch.no_grad():
                _ = trt_model(example_input)

            self.logger.info("✅ TensorRT optimization successful")
            return trt_model

        except Exception as e:
            self.logger.warning(f"TensorRT optimization failed: {e}")
            self.logger.info("Continuing with standard PyTorch model")
            return model

    def optimize_cuda_memory(self):
        """Optimize CUDA memory usage"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Set memory fraction (use 80% of GPU memory)
            torch.cuda.set_per_process_memory_fraction(0.8)

            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass

            self.logger.info("CUDA memory optimized")

    def load_model(self):
        """Load trained model (supports both transformer and feedforward architectures)"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Extract model configuration
            config = checkpoint.get('model_config', {}) or {}
            state_dict = checkpoint['model_state_dict']
            arch = config.get('arch')
            if arch is None:
                if 'hidden_sizes' in config:
                    arch = 'feedforward'
                elif 'resfeedforward' in config.get('model_name', '').lower():
                    arch = 'resfeedforward'
                else:
                    arch = 'transformer'

            arch_lower = arch.lower()

            # Create model based on architecture
            if 'feedforward' in arch_lower and 'res' not in arch_lower:
                hidden_sizes = config.get('hidden_sizes')
                has_layer_norm = any('.1.' in key for key in state_dict.keys())

                if not hidden_sizes and not has_layer_norm:
                    # Legacy 2-layer feedforward (Linear-ReLU-Linear)
                    self.model = SimpleTransformer(
                        input_dim=4,
                        output_dim=4,
                        d_model=config.get('d_model', config.get('dim_feedforward', 12)),
                        nhead=config.get('nhead', 1),
                        num_layers=1,
                        dim_feedforward=config.get('dim_feedforward', 12),
                        dropout=0.0
                    ).to(self.device)
                    self.logger.info("Detected legacy feedforward checkpoint; using SimpleTransformer layout")
                else:
                    if not hidden_sizes:
                        num_layers = max(int(config.get('num_layers', 1)), 1)
                        dim_size = int(config.get('dim_feedforward', 12))
                        hidden_sizes = tuple(dim_size for _ in range(num_layers))
                    else:
                        hidden_sizes = tuple(int(size) for size in hidden_sizes)

                    dropout = float(config.get('dropout', 0.0))
                    self.model = ConfigurableFeedforward(
                        input_dim=4,
                        output_dim=4,
                        hidden_sizes=hidden_sizes,
                        dropout=dropout
                    ).to(self.device)
                    self.logger.info(
                        "Created ConfigurableFeedforward model with hidden_sizes=%s, dropout=%.3f",
                        hidden_sizes,
                        dropout
                    )

            elif 'res' in arch_lower:
                # Residual feedforward model
                self.model = ResFeedforward(
                    input_dim=4,
                    output_dim=4,
                    dropout=config.get('dropout', 0.0)
                ).to(self.device)
                self.logger.info(f"Created ResFeedforward model: {arch}")

            else:
                # Default transformer model
                self.model = SimpleTransformer(
                    input_dim=4,
                    output_dim=4,
                    d_model=config.get('d_model', 8),
                    nhead=config.get('nhead', 1),
                    num_layers=config.get('num_layers', 1),
                    dim_feedforward=config.get('dim_feedforward', 12),
                    dropout=0.0
                ).to(self.device)
                self.logger.info("Created SimpleTransformer model")

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Apply TensorRT optimization if available
            if self.use_tensorrt and torch.cuda.is_available():
                self.model = self._optimize_with_tensorrt(self.model)

            # Load scalers
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            self.normalize = checkpoint.get('normalize', True)

            self.logger.info(f"Model loaded from {self.model_path} (arch: {arch})")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def load_hardware_config(self, config_path):
        """Load hardware configuration"""
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

    def _camera_capture_thread(self, camera_name, camera):
        """Continuous camera capture in separate thread"""
        while self.capture_active:
            try:
                ret, frame = camera.read()
                if ret:
                    # Apply color conversion if needed
                    converter = self.camera_color_converters.get(camera_name)
                    if converter is not None:
                        try:
                            frame = cv2.cvtColor(frame, converter)
                        except cv2.error as error:
                            self.logger.warning(f"Color conversion failed for {camera_name} camera: {error}")
                            self.camera_color_converters[camera_name] = None

                    # Put frame in queue (replace old frame if queue is full)
                    try:
                        self.frame_queues[camera_name].put_nowait(frame)
                    except:
                        # Queue full, remove old frame and add new one
                        try:
                            self.frame_queues[camera_name].get_nowait()
                            self.frame_queues[camera_name].put_nowait(frame)
                        except Empty:
                            pass
                else:
                    # Camera read failed, put None
                    try:
                        self.frame_queues[camera_name].put_nowait(None)
                    except:
                        pass

            except Exception as e:
                self.logger.error(f"Camera {camera_name} capture error: {e}")
                time.sleep(0.001)  # Brief pause on error

    def start_camera_threads(self):
        """Start threaded camera capture"""
        self.capture_active = True

        for camera_name, camera in self.cameras.items():
            if camera is not None:
                # Create queue for this camera (size 2 to prevent lag)
                self.frame_queues[camera_name] = Queue(maxsize=2)

                # Start capture thread
                thread = threading.Thread(
                    target=self._camera_capture_thread,
                    args=(camera_name, camera),
                    daemon=True
                )
                thread.start()
                self.camera_threads[camera_name] = thread
                self.logger.info(f"Started capture thread for {camera_name} camera")

    def stop_camera_threads(self):
        """Stop threaded camera capture"""
        self.capture_active = False

        # Wait for threads to finish
        for camera_name, thread in self.camera_threads.items():
            thread.join(timeout=1.0)
            self.logger.info(f"Stopped capture thread for {camera_name} camera")

        self.camera_threads.clear()

    def get_latest_frames(self):
        """Get latest frames from all cameras (non-blocking)"""
        frames = {}

        for camera_name in self.cameras.keys():
            try:
                # Get most recent frame (non-blocking)
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
        """Setup camera capture with Jetson GStreamer optimization"""
        self.cameras = {}
        camera_config = self.config.get('cameras', {})

        # Left camera setup
        left_config = camera_config.get('cam_left', {'id': 0, 'enabled': True})
        if left_config.get('enabled', True):
            left_id = left_config.get('id', 0)
            self.cameras['left'] = self._setup_single_camera('left', left_id, left_config)
        else:
            self.cameras['left'] = None
            self.logger.info("Left camera disabled in configuration")

        # Right camera setup
        right_config = camera_config.get('cam_right', {'id': 2, 'enabled': True})
        if right_config.get('enabled', True):
            right_id = right_config.get('id', 2)
            self.cameras['right'] = self._setup_single_camera('right', right_id, right_config)
        else:
            self.cameras['right'] = None
            self.logger.info("Right camera disabled in configuration")

    def _setup_single_camera(self, camera_name, camera_id, config):
        """Setup single camera with GStreamer optimization for Jetson"""
        camera = None

        # Try CSI camera first if on Jetson
        if hasattr(self, 'is_jetson') and self.is_jetson and camera_id in [0, 1]:
            try:
                # GStreamer pipeline for CSI camera (ultra-fast)
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

        # Fallback to USB with V4L2 optimization
        try:
            # Try V4L2 backend first for better performance
            camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            if not camera.isOpened():
                camera = cv2.VideoCapture(camera_id)

            if camera.isOpened():
                # Apply maximum-speed optimizations
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 120)

                # Apply FOURCC optimization - use MJPG for best performance (no conversion needed)
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
        """Apply requested FOURCC and track any needed color conversion."""
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
        """Return the active FOURCC string for the given camera."""
        try:
            value = int(camera.get(cv2.CAP_PROP_FOURCC))
        except Exception:
            return ''

        if value == 0:
            return ''

        chars = [chr((value >> (8 * i)) & 0xFF) for i in range(4)]
        return ''.join(chars).strip()

    def _color_converter_for_fourcc(self, fourcc_code):
        """Determine if manual color conversion is needed for the FOURCC."""
        if not fourcc_code:
            return None

        code = fourcc_code.upper()
        # Only convert if camera outputs YUV formats
        if code in {'YUYV', 'YUY2', 'YUNV'}:
            return getattr(cv2, 'COLOR_YUV2BGR_YUY2', None)
        if code in {'UYVY', 'YVYU'}:
            return getattr(cv2, 'COLOR_YUV2BGR_UYVY', None)
        # BGR and MJPG formats need no conversion - major performance boost!
        if code in {'BGR3', 'BGR4', 'BGRA', 'RGB3', 'RGB4', 'RGBA', 'MJPG'}:
            return None
        return None


    def setup_display_windows(self):
        """Create OpenCV windows when display mode is enabled."""
        # Only create windows for available cameras
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
        """Render camera frames with hand detection overlays and coordinates."""
        if not self.show_display:
            return

        # Process left camera
        left_frame = frames.get('left')
        if left_frame is not None:
            display_frame = left_frame.copy()
            x, y = left_features

            # Add camera info overlay
            self._add_camera_overlay(display_frame, 'LEFT', x, y, left_features, right_features)

            # Draw hand detection if coordinates are valid
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and (x != 0 or y != 0):
                # Draw index finger tip
                cv2.circle(display_frame, (int(x), int(y)), 12, (0, 255, 0), -1)
                cv2.circle(display_frame, (int(x), int(y)), 15, (255, 255, 255), 2)

                # Draw coordinate text with background
                coord_text = f"Index: ({int(x)}, {int(y)})"
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = int(x) + 20, int(y) - 20

                # Ensure text stays within frame
                if text_x + text_size[0] > display_frame.shape[1]:
                    text_x = int(x) - text_size[0] - 20
                if text_y - text_size[1] < 0:
                    text_y = int(y) + 40

                # Draw text background
                cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, coord_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Left Camera", display_frame)

        # Process right camera
        right_frame = frames.get('right')
        if right_frame is not None:
            display_frame = right_frame.copy()
            x, y = right_features

            # Add camera info overlay
            self._add_camera_overlay(display_frame, 'RIGHT', x, y, left_features, right_features)

            # Draw hand detection if coordinates are valid
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and (x != 0 or y != 0):
                # Draw index finger tip
                cv2.circle(display_frame, (int(x), int(y)), 12, (0, 255, 255), -1)  # Yellow for right
                cv2.circle(display_frame, (int(x), int(y)), 15, (255, 255, 255), 2)

                # Draw coordinate text with background
                coord_text = f"Index: ({int(x)}, {int(y)})"
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = int(x) + 20, int(y) - 20

                # Ensure text stays within frame
                if text_x + text_size[0] > display_frame.shape[1]:
                    text_x = int(x) - text_size[0] - 20
                if text_y - text_size[1] < 0:
                    text_y = int(y) + 40

                # Draw text background
                cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, coord_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Right Camera", display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord('s'):
            self.logger.warning('Emergency stop triggered via display window')
            self.emergency_stop = True

    def _add_camera_overlay(self, frame, camera_name, x, y, left_coords, right_coords):
        """Add camera information overlay to the frame"""
        h, w = frame.shape[:2]

        # Status color based on hand detection
        hand_detected = isinstance(x, (int, float)) and isinstance(y, (int, float)) and (x != 0 or y != 0)
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if hand_detected else "NO HAND"

        # Draw background rectangle for overlay (larger for stereo info)
        overlay_height = 140
        cv2.rectangle(frame, (10, 10), (350, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, overlay_height), status_color, 2)

        # Camera name
        cv2.putText(frame, f"{camera_name} CAMERA", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Status
        cv2.putText(frame, f"Status: {status_text}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # This camera coordinates
        if hand_detected:
            cv2.putText(frame, f"This: ({int(x)}, {int(y)})", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "This: (---, ---)", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

        # Stereo coordinates (both cameras)
        left_x, left_y = left_coords
        right_x, right_y = right_coords

        left_valid = isinstance(left_x, (int, float)) and isinstance(left_y, (int, float)) and (left_x != 0 or left_y != 0)
        right_valid = isinstance(right_x, (int, float)) and isinstance(right_y, (int, float)) and (right_x != 0 or right_y != 0)

        left_text = f"L:({int(left_x)},{int(left_y)})" if left_valid else "L:(---,---)"
        right_text = f"R:({int(right_x)},{int(right_y)})" if right_valid else "R:(---,---)"

        cv2.putText(frame, f"Stereo: {left_text} | {right_text}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # Instructions at bottom
        cv2.putText(frame, "Press 'q' to quit, 's' for emergency stop",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def setup_mediapipe(self):
        """Setup separate MediaPipe instances for left and right cameras"""
        self.mp_hands = mp.solutions.hands

        # Separate MediaPipe instances for each camera
        self.hands_left = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lowered for faster detection
            min_tracking_confidence=0.2,   # Lowered for faster tracking
            model_complexity=0              # Fastest model
        )

        self.hands_right = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lowered for faster detection
            min_tracking_confidence=0.2,   # Lowered for faster tracking
            model_complexity=0              # Fastest model
        )

        self.mp_drawing = mp.solutions.drawing_utils

    def setup_servo_defaults(self):
        """Setup default servo parameters for test mode"""
        robot_config = self.config.get('robot_arms', {})
        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4])
        self.min_positions = [1024, 1024, 1024, 1024]  # Default mins
        self.max_positions = [2944, 3456, 3200, 3136]  # Default maxs

    def setup_servos(self):
        """Setup servo communication"""
        if not DynamixelSDK_available:
            return

        robot_config = self.config.get('robot_arms', {})
        follower_config = robot_config.get('follower', {})

        # Initialize port
        port = follower_config.get('port', '/dev/follower_arm')
        baudrate = follower_config.get('baudrate', 1000000)

        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(robot_config.get('protocol_version', 2.0))

        # Open port
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {port}")

        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            raise Exception(f"Failed to set baudrate {baudrate}")

        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4])
        self.min_positions = [1280, 1920, 1120, 1664]  # Default safe range
        self.max_positions = [3072, 3072, 3072, 3072]  # Default safe range

        # Setup GroupSyncWrite for position control
        goal_position_addr = robot_config.get('addr_goal_position', 116)
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, goal_position_addr, 4)

        # Enable torque for all servos
        self.enable_torque()

        self.logger.info(f"Servos initialized on {port}")

    def enable_torque(self):
        """Enable torque for all servos"""
        if not DynamixelSDK_available:
            return

        torque_enable_addr = 64  # ADDR_TORQUE_ENABLE for XM series

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
        """Disable torque for all servos"""
        if not DynamixelSDK_available:
            return

        torque_enable_addr = 64  # ADDR_TORQUE_ENABLE for XM series

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
        """Extract hand landmark features from camera frame with separate MediaPipe instances"""
        if frame is None:
            return [0.0, 0.0], None  # Return coordinates and processed frame

        try:
            # Validate frame
            if frame.size == 0 or len(frame.shape) != 3:
                self.logger.warning(f"Invalid frame from {camera_name}")
                return [0.0, 0.0], frame.copy()

            # Select appropriate MediaPipe instance
            hands_processor = self.hands_left if camera_name == 'left' else self.hands_right

            # Optimize MediaPipe processing
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_processor.process(rgb_frame)
            frame.flags.writeable = True

            # Create display frame with landmarks
            display_frame = frame.copy()

            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw hand landmarks on display frame
                self.mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get index finger tip (landmark 8)
                index_tip = hand_landmarks.landmark[8]

                # Validate landmark coordinates
                if 0 <= index_tip.x <= 1 and 0 <= index_tip.y <= 1:
                    # Convert normalized coordinates to pixel coordinates
                    h, w = frame.shape[:2]
                    x = index_tip.x * w
                    y = index_tip.y * h

                    # Sanity check coordinates
                    if 0 <= x <= w and 0 <= y <= h:
                        return [x, y], display_frame

            return [0.0, 0.0], display_frame  # Default if no valid hand detected

        except Exception as e:
            self.logger.error(f"Hand detection error in {camera_name}: {e}")
            return [0.0, 0.0], frame.copy()

    def predict_joint_positions(self, features):
        """Predict joint positions using transformer model with error handling"""
        try:
            # Validate input features
            features = np.array(features, dtype=np.float32)
            if not np.isfinite(features).all():
                self.logger.warning("Non-finite features detected, using defaults")
                return self.last_successful_positions.copy()

            features = features.reshape(1, -1)

            # Normalize if required
            if self.normalize:
                features = self.scaler_X.transform(features)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)

            # Predict
            with torch.no_grad():
                predictions = self.model(features_tensor).cpu().numpy()

            # Denormalize if required
            if self.normalize:
                predictions = self.scaler_y.inverse_transform(predictions)

            result = predictions[0]

            # Validate predictions
            if not np.isfinite(result).all():
                self.logger.warning("Non-finite predictions, using last successful positions")
                return self.last_successful_positions.copy()

            # Clamp to safe zone ranges
            for i in range(len(result)):
                min_safe = self.safe_zone_min[i] if i < len(self.safe_zone_min) else 1200
                max_safe = self.safe_zone_max[i] if i < len(self.safe_zone_max) else 2896

                if result[i] < min_safe or result[i] > max_safe:
                    # Only log if significantly out of safe range
                    if result[i] < min_safe - 200 or result[i] > max_safe + 200:
                        self.logger.warning(f"Joint {i} prediction {result[i]:.0f} clamped to safe range [{min_safe}-{max_safe}]")
                    result[i] = max(min_safe, min(max_safe, result[i]))

            return result

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.last_successful_positions.copy()

    def clamp_positions(self, positions):
        """Apply safety constraints and smooth position changes"""
        if self.emergency_stop:
            self.logger.warning("Emergency stop active - holding position")
            return self.last_positions.copy()

        safe_positions = []
        for i, pos in enumerate(positions):
            # Use conservative safe limits
            min_pos = self.safe_zone_min[i]
            max_pos = self.safe_zone_max[i]

            # Clamp to safe zone
            safe_pos = max(min_pos, min(max_pos, int(pos)))

            # Apply rate limiting for safety
            if i < len(self.last_positions):
                last_pos = self.last_positions[i]

                # No per-iteration rate limiting - allow immediate position changes

                # Apply smoothing for stable movement
                smoothed_pos = (self.position_smoothing_alpha * safe_pos +
                               (1 - self.position_smoothing_alpha) * last_pos)
                safe_pos = int(smoothed_pos)

            safe_positions.append(safe_pos)

        # Update last positions
        self.last_positions = safe_positions.copy()

        # Check for dangerous movements
        self.check_safety_violations(safe_positions)

        return safe_positions


    def check_safety_violations(self, positions):
        """Check for safety violations and trigger emergency stop if needed"""
        for i, pos in enumerate(positions):
            # Check if position is outside safe zone
            min_safe = self.safe_zone_min[i] if i < len(self.safe_zone_min) else 1200
            max_safe = self.safe_zone_max[i] if i < len(self.safe_zone_max) else 2896

            if pos < min_safe or pos > max_safe:
                self.logger.error(f"Safety violation: Joint {i} at position {pos}, safe range [{min_safe}-{max_safe}]")
                self.emergency_stop = True
                return

        # Reset emergency stop if positions are safe
        if self.emergency_stop:
            all_safe = all(self.safe_zone_min[i] <= pos <= self.safe_zone_max[i]
                          for i, pos in enumerate(positions) if i < len(self.safe_zone_min))
            if all_safe:
                self.logger.info("Positions back in safe zone, resetting emergency stop")
                self.emergency_stop = False

    def send_servo_commands(self, positions):
        """Send position commands to servos with safety checks"""
        if self.test_mode or not DynamixelSDK_available:
            return True

        if self.emergency_stop:
            self.logger.warning("Emergency stop active - not sending commands")
            return False

        try:
            # Final safety check before sending
            for i, pos in enumerate(positions):
                min_safe = self.safe_zone_min[i] if i < len(self.safe_zone_min) else 1200
                max_safe = self.safe_zone_max[i] if i < len(self.safe_zone_max) else 2896

                if pos < min_safe or pos > max_safe:
                    self.logger.error(f"Refusing to send unsafe position {pos} to joint {i}, safe range [{min_safe}-{max_safe}]")
                    self.emergency_stop = True
                    return False

            # Clear previous data
            self.group_sync_write.clearParam()

            # Add each servo position
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

            # Send command
            dxl_comm_result = self.group_sync_write.txPacket()
            success = dxl_comm_result == COMM_SUCCESS

            if success:
                self.last_successful_positions = positions.copy()
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.logger.warning(f"Servo command failed, consecutive failures: {self.consecutive_failures}")

                # Emergency stop if too many failures
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

    def run_control_loop(self):
        """Main control loop"""
        self.logger.info(f"🧪 TEST Starting control loop at {self.target_fps:.2f} Hz" if self.test_mode
                        else f"🤖 Starting control loop at {self.target_fps:.2f} Hz")

        try:
            while True:
                loop_start = time.time()

                # Get latest frames from threaded capture (much faster!)
                frames = self.get_latest_frames()

                # Extract features from both cameras (now returns coordinates and processed frames)
                left_features, left_processed_frame = self.extract_hand_features(frames.get('left'), 'left')
                right_features, right_processed_frame = self.extract_hand_features(frames.get('right'), 'right')

                # Update frames with processed versions for display
                if left_processed_frame is not None:
                    frames['left'] = left_processed_frame
                if right_processed_frame is not None:
                    frames['right'] = right_processed_frame

                if self.show_display:
                    self.update_display(frames, left_features, right_features)

                # Combine features: [left_x, left_y, right_x, right_y]
                combined_features = left_features + right_features

                # Predict joint positions
                inference_start = time.time()
                predicted_positions = self.predict_joint_positions(combined_features)
                inference_time = time.time() - inference_start

                # Clamp positions to valid range
                clamped_positions = self.clamp_positions(predicted_positions)

                # Send commands to servos with safety check
                if not self.test_mode:
                    if not self.send_servo_commands(clamped_positions):
                        self.logger.warning("Failed to send servo commands, using safe position")
                        # Return to safe position on failure
                        safe_positions = [2048, 3328, 1140, 3072]
                        self.send_servo_commands(safe_positions)

                # Statistics
                self.frame_count += 1
                self.total_inference_time += inference_time

                # Log periodically
                if self.frame_count % 120 == 0:  # Every 2 seconds at 60fps
                    current_time = time.time()
                    elapsed = current_time - self.last_fps_time
                    fps = 120 / elapsed
                    avg_inference = (self.total_inference_time / self.frame_count) * 1000

                    mode_prefix = "🧪" if self.test_mode else "🤖"
                    self.logger.info(f"{mode_prefix} FPS: {fps:.1f}, Avg inference: {avg_inference:.1f}ms")
                    self.last_fps_time = current_time

                # Test mode logging
                if self.test_mode and self.frame_count % 120 == 1:
                    self.logger.info(f"🧪 Features: {combined_features} → Predicted: {clamped_positions}")

                # No frame rate limiting - run at maximum speed

        except KeyboardInterrupt:
            self.logger.info("Control loop stopped by user")
            # Stop camera threads
            self.stop_camera_threads()
            # Return to safe position before stopping
            if not self.test_mode:
                safe_positions = [2048, 3328, 1140, 3072]
                self.send_servo_commands(safe_positions)
                self.logger.info("Returned to safe position")
        except Exception as e:
            self.logger.error(f"Control loop error: {e}")
            # Stop camera threads
            self.stop_camera_threads()
            # Emergency safe position
            if not self.test_mode:
                try:
                    safe_positions = [2048, 3328, 1140, 3072]
                    self.send_servo_commands(safe_positions)
                    self.logger.info("Emergency return to safe position")
                except:
                    self.logger.error("Failed to return to safe position")
            raise

    def cleanup(self):
        """Cleanup resources safely"""
        # Return to safe position first
        if not self.test_mode and DynamixelSDK_available:
            try:
                safe_positions = [2048, 3328, 1140, 3072]
                self.send_servo_commands(safe_positions)
                time.sleep(0.5)  # Give time to reach safe position
                self.logger.info("Returned to safe position during cleanup")
            except Exception as e:
                self.logger.error(f"Failed to return to safe position: {e}")

        # Disable torque
        if not self.test_mode and DynamixelSDK_available:
            try:
                self.disable_torque()
            except Exception as e:
                self.logger.error(f"Failed to disable torque: {e}")

        # Close cameras
        for camera in self.cameras.values():
            if camera is not None:
                try:
                    camera.release()
                except Exception as e:
                    self.logger.error(f"Failed to release camera: {e}")

        if self.show_display:
            cv2.destroyAllWindows()

        # Close serial port
        if hasattr(self, 'port_handler') and not self.test_mode:
            try:
                self.port_handler.closePort()
            except Exception as e:
                self.logger.error(f"Failed to close port: {e}")

        self.logger.info("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="Transformer Hardware Runner")
    parser.add_argument("--model", required=True, help="Path to trained transformer model (.pth)")
    parser.add_argument("--config", default="hardware_config.json", help="Hardware configuration file")
    parser.add_argument("--test", action='store_true', help="Run in test mode (no hardware control)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS (default: 60.0)")
    parser.add_argument("--display", action='store_true', help="Show OpenCV preview windows")
    parser.add_argument("--no-tensorrt", action='store_true', help="Disable TensorRT optimization")

    args = parser.parse_args()

    print("🤖 Transformer Hardware Runner")
    print("=" * 50)

    try:
        # Initialize runner
        runner = TransformerHardwareRunner(
            model_path=args.model,
            hardware_config_path=args.config,
            test_mode=args.test,
            target_fps=args.fps,
            show_display=args.display,
            use_tensorrt=not args.no_tensorrt
        )

        # Run control loop
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
