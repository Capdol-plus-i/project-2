#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Voice + Hardware Runner
Combines XGBoost/PyTorch camera-to-joint mapping with voice control for Arduino
- Camera mode: Real-time hand tracking → robot arm control
- Voice mode: Speech recognition → Arduino LED control
- Switch between modes with voice commands or keyboard
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
import joblib
import re
import unicodedata
import queue
import audioop
import ctypes
import ctypes.util
import struct
from typing import Optional, List, Tuple

# Hardware imports
try:
    from dynamixel_sdk import *
except ImportError:
    print("⚠️ DynamixelSDK not available - running in test mode only")
    DynamixelSDK_available = False
else:
    DynamixelSDK_available = True

# Voice control imports
import serial
import pyaudio
import webrtcvad
from google.cloud import speech

# Jetson optimization imports
try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Suppress audio/logging noise
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ.setdefault("ALSA_LOG_LEVEL", "0")

try:
    _asound = ctypes.CDLL(ctypes.util.find_library("asound"))
    CMPFUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                               ctypes.c_int, ctypes.c_char_p)
    def _py_alsa_err_handler(filename, line, function, err, fmt): return
    _c_err_handler = CMPFUNC(_py_alsa_err_handler)
    _asound.snd_lib_error_set_handler(_c_err_handler)
except Exception:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

# Audio settings
TARGET_RATE = 16000
FRAME_MS = 10
BYTES_PER_SAMPLE = 2
SAMPLES_PER_FRAME = int(TARGET_RATE * FRAME_MS / 1000)
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE
DEFAULT_MIC_HINT = "blue"

# Voice commands
WAKE_CANONICAL = "하이봇"
WAKE_VARIANTS = [
    "하이봇", "하이 봇", "하이봇아", "하 이 봇",
    "아이봇", "하이보", "하이 보트"
]

COMMAND_SYNONYMS = {
    "OFF": ["꺼", "꺼줘", "불꺼", "불 꺼", "라이트오프", "라이트 오프", "끄자"],
    "ON": ["켜", "켜줘", "불켜", "불 켜", "라이트온", "라이트 온", "키자"],
    "CAMERA_MODE": ["카메라", "카메라 모드", "손", "손 모드", "캠", "핸드"],
    "VOICE_MODE": ["음성", "음성 모드", "보이스", "보이스 모드", "말"],
    "EXIT": ["종료", "끝내", "그만", "나가", "종 료"]
}

# =============================================================================
# PYTORCH MODEL CLASSES
# =============================================================================

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

# =============================================================================
# VOICE RECOGNITION UTILITIES
# =============================================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).lower()

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost))
        prev = cur
    return prev[-1]

def fuzzy_match_word(text: str, target: str, max_dist: int) -> bool:
    t = normalize(target)
    s = normalize(text)
    n = len(t)
    if n == 0: return False
    if len(s) < n: return levenshtein(s, t) <= max_dist
    if t in s: return True
    for i in range(len(s) - n + 1):
        if levenshtein(s[i:i+n], t) <= max_dist:
            return True
    return False

def is_wake_word(text: str) -> bool:
    if fuzzy_match_word(text, WAKE_CANONICAL, 1): return True
    for w in WAKE_VARIANTS:
        tol = 1 if len(w) <= 3 else 2
        if fuzzy_match_word(text, w, tol): return True
    return False

def which_command(text: str) -> Optional[str]:
    s = normalize(text)
    for cmd, syns in COMMAND_SYNONYMS.items():
        for k in syns:
            kn = normalize(k)
            if len(kn) <= 2 and kn in s:
                return cmd
    for cmd, syns in COMMAND_SYNONYMS.items():
        for k in syns:
            kn = normalize(k)
            if len(kn) >= 3 and kn in s:
                return cmd
    for cmd, syns in COMMAND_SYNONYMS.items():
        for k in syns:
            kn = normalize(k)
            if len(kn) >= 3 and fuzzy_match_word(s, kn, 1):
                return cmd
    return None

def quick_contains(text: str, keys: List[str]) -> bool:
    s = normalize(text)
    return any(normalize(k) in s for k in keys)

def detect_wake_interim(text: str) -> bool:
    keys = [WAKE_CANONICAL] + WAKE_VARIANTS
    return quick_contains(text, keys)

def detect_cmd_interim(text: str) -> Optional[str]:
    # Priority order
    if quick_contains(text, COMMAND_SYNONYMS["EXIT"]): return "EXIT"
    if quick_contains(text, COMMAND_SYNONYMS["CAMERA_MODE"]): return "CAMERA_MODE"
    if quick_contains(text, COMMAND_SYNONYMS["VOICE_MODE"]): return "VOICE_MODE"
    if quick_contains(text, COMMAND_SYNONYMS["OFF"]): return "OFF"
    if quick_contains(text, COMMAND_SYNONYMS["ON"]): return "ON"
    return None

# =============================================================================
# MICROPHONE STREAM
# =============================================================================

def list_input_devices():
    p = pyaudio.PyAudio()
    print("=== Input devices ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            print(f"[{i}] {info.get('name')}  (in={info.get('maxInputChannels')}, rate={int(info.get('defaultSampleRate',0))})")
    p.terminate()

def pick_device_index(p: pyaudio.PyAudio, index: Optional[int], hint: str) -> int:
    if index is not None:
        return index
    chosen = None
    hint_l = (hint or "").lower()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        name = (info.get("name", "") or "").lower()
        if hint_l and hint_l in name:
            return i
        if chosen is None:
            chosen = i
    return chosen if chosen is not None else 0

class MicrophoneStream:
    def __init__(self, mic_index: Optional[int], mic_hint: str, debug: bool):
        self.mic_index = mic_index
        self.mic_hint = mic_hint
        self.debug = debug
        self._pa = None
        self._stream = None
        self._buff = queue.Queue(maxsize=100)
        self._carry = b""
        self._ratecv_state = None
        self._hw_rate = None
        self._hw_channels = 1
        self.closed = True
        self.vad = webrtcvad.Vad(1)

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        device_index = pick_device_index(self._pa, self.mic_index, self.mic_hint)
        dinfo = self._pa.get_device_info_by_index(device_index)
        default_rate = int(dinfo.get("defaultSampleRate", 48000))
        rate_candidates = [16000, default_rate, 48000, 44100, 32000]
        last_err = None
        for ch in (1, 2):
            for r in rate_candidates:
                try:
                    frames_per_buffer = int(r * FRAME_MS / 1000)
                    self._stream = self._pa.open(
                        format=pyaudio.paInt16,
                        channels=ch, rate=r, input=True,
                        input_device_index=device_index,
                        frames_per_buffer=frames_per_buffer,
                        stream_callback=self._fill_buffer,
                    )
                    self._hw_rate, self._hw_channels = r, ch
                    self.closed = False
                    print(f"🎤 Mic: [{device_index}] {dinfo.get('name')} @ {r} Hz, ch={ch}")
                    return self
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"마이크 열기 실패: {last_err}")

    def __exit__(self, *args):
        self.closed = True
        if self._stream:
            try: self._stream.stop_stream()
            except: pass
            try: self._stream.close()
            except: pass
        try: self._buff.put_nowait(None)
        except: pass
        if self._pa: self._pa.terminate()

    def _fill_buffer(self, in_data, *_):
        try:
            if self._buff.full(): self._buff.get_nowait()
            self._buff.put_nowait(in_data)
        except queue.Full: pass
        return (None, pyaudio.paContinue)

    def _to_mono_16k(self, data: bytes) -> bytes:
        pcm = data
        if self._hw_channels == 2:
            try:
                pcm = audioop.tomono(pcm, BYTES_PER_SAMPLE, 0.5, 0.5)
            except Exception:
                mono = bytearray()
                for (l, r) in struct.iter_unpack('<hh', pcm):
                    mono.extend(struct.pack('<h', int((l + r)/2)))
                pcm = bytes(mono)
        if self._hw_rate != TARGET_RATE:
            pcm, self._ratecv_state = audioop.ratecv(
                pcm, BYTES_PER_SAMPLE, 1,
                self._hw_rate, TARGET_RATE, self._ratecv_state
            )
        return pcm

    def generator(self):
        while not self.closed:
            try:
                chunk = self._buff.get(timeout=1.0)
            except queue.Empty:
                continue
            if chunk is None:
                return
            pcm16k = self._to_mono_16k(chunk)
            if self.debug:
                try:
                    voiced = self.vad.is_speech(pcm16k[:FRAME_BYTES], TARGET_RATE)
                    print(f"\r[VAD] {'speech' if voiced else 'silence'}", end="", flush=True)
                except:
                    pass
            yield pcm16k

# =============================================================================
# GOOGLE SPEECH CLIENT
# =============================================================================

def build_client_and_config(single_utter: bool):
    client = speech.SpeechClient()
    phrases = list(set(
        [WAKE_CANONICAL] + WAKE_VARIANTS +
        ["불 꺼 줘", "불켜 줘", "라이트 오프", "라이트 온",
         "라이트오프", "라이트온", "종료해", "끝내", "그만",
         "카메라", "카메라 모드", "음성", "음성 모드"] +
        sum(COMMAND_SYNONYMS.values(), [])
    ))
    speech_context = speech.SpeechContext(phrases=phrases, boost=20.0)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_RATE,
        language_code="ko-KR",
        speech_contexts=[speech_context],
        max_alternatives=3,
        enable_automatic_punctuation=False,
        use_enhanced=True,
        model="command_and_search",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=single_utter
    )
    return client, streaming_config

def start_stream(mic_index: Optional[int], mic_hint: str, debug: bool, for_command: bool):
    client, streaming_config = build_client_and_config(single_utter=for_command)
    stream = MicrophoneStream(mic_index, mic_hint, debug)
    stream.__enter__()
    audio_gen = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=f) for f in audio_gen)
    responses = client.streaming_recognize(streaming_config, requests)
    return stream, responses

# =============================================================================
# ARDUINO INTERFACE
# =============================================================================

def open_arduino(port: str, baud: int = 9600):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"✓ Arduino 연결 성공! ({port})")
        return ser
    except Exception as e:
        print(f"❌ Arduino 연결 실패: {e}")
        return None

def send_arduino(ser, cmd: str):
    try:
        ser.write((cmd + "\n").encode())
        ser.flush()
        print(f"👉 Arduino: {cmd}")
    except Exception as e:
        print(f"❌ Arduino 전송 실패: {e}")

# =============================================================================
# MAIN UNIFIED HARDWARE RUNNER
# =============================================================================

class UnifiedVoiceHardwareRunner:
    def __init__(self, model_path=None, hardware_config_path='hardware_config.json',
                 arduino_port="/dev/arduino", test_mode=False, target_fps=60.0,
                 show_display=False, mic_index=None, mic_hint="blue", debug=False):

        # Mode control
        self.current_mode = "camera"  # "camera" or "voice"
        self.mode_switching = False
        self.running = True

        # Hardware runner setup
        self.model_path = model_path
        self.test_mode = test_mode
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.show_display = show_display

        # Voice control setup
        self.arduino_port = arduino_port
        self.mic_index = mic_index
        self.mic_hint = mic_hint
        self.debug = debug
        self.arduino = None

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load model if provided
        self.model = None
        self.model_type = None
        self.scaler_X = None
        self.scaler_y = None
        self.normalize = False

        if model_path:
            self.load_model()

        # Load hardware configuration
        self.load_hardware_config(hardware_config_path)

        # Initialize camera system
        self.frame_queues = {}
        self.camera_threads = {}
        self.capture_active = False
        self.cameras = {}
        self.camera_color_converters = {}

        self.setup_cameras()
        self.setup_mediapipe()
        if self.show_display:
            self.setup_display_windows()

        # Initialize servos if not in test mode
        if not test_mode and DynamixelSDK_available:
            self.setup_servos()
        else:
            self.logger.info("🧪 TEST MODE - Hardware control disabled")
            self.setup_servo_defaults()

        # Safety and control variables
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.last_fps_time = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_successful_positions = [2048, 3328, 1140, 3072]
        self.position_smoothing_alpha = 0.2
        self.last_positions = [2048, 3328, 1140, 3072]
        self.emergency_stop = False
        self.safe_zone_min = [1280, 1920, 1120, 1664]
        self.safe_zone_max = [2944, 3456, 3200, 3136]

        # Initialize Arduino
        self.arduino = open_arduino(self.arduino_port)

    def load_model(self):
        """Load XGBoost or PyTorch model"""
        self.model_type = None
        self.normalize = False
        self.scaler_X = None
        self.scaler_y = None

        if self.model_path.lower().endswith(".joblib"):
            self.logger.info(f"Loading XGBoost model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_type = "xgb"
            self.logger.info("XGBoost model loaded (NaN inputs supported).")
            return

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

    def setup_cameras(self):
        """Setup camera system"""
        self.cameras = {}
        camera_config = self.config.get('cameras', {})

        left_config = camera_config.get('cam_left', {'id': 0, 'enabled': True})
        if left_config.get('enabled', True):
            left_id = left_config.get('id', 0)
            self.cameras['left'] = self._setup_single_camera('left', left_id, left_config)
        else:
            self.cameras['left'] = None

        right_config = camera_config.get('cam_right', {'id': 2, 'enabled': True})
        if right_config.get('enabled', True):
            right_id = right_config.get('id', 2)
            self.cameras['right'] = self._setup_single_camera('right', right_id, right_config)
        else:
            self.cameras['right'] = None

    def _setup_single_camera(self, camera_name, camera_id, config):
        """Setup single camera"""
        try:
            camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            if not camera.isOpened():
                camera = cv2.VideoCapture(camera_id)

            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.logger.info(f"{camera_name.title()} camera ready (id={camera_id})")
                return camera
            else:
                self.logger.warning(f"{camera_name.title()} camera not available (id={camera_id})")
                return None
        except Exception as e:
            self.logger.error(f"Failed to setup {camera_name} camera: {e}")
            return None

    def setup_mediapipe(self):
        """Setup MediaPipe hand tracking"""
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

    def setup_display_windows(self):
        """Setup OpenCV display windows"""
        if self.cameras.get('left') is not None:
            cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Left Camera', 640, 480)
            cv2.moveWindow('Left Camera', 100, 100)
        if self.cameras.get('right') is not None:
            cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Right Camera', 640, 480)
            cv2.moveWindow('Right Camera', 780, 100)
        self.logger.info("Display windows initialized - press 'q' to quit, 'v' for voice mode, 'c' for camera mode")

    def setup_servo_defaults(self):
        """Setup default servo parameters"""
        robot_config = self.config.get('robot_arms', {})
        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4])
        self.min_positions = [1024, 1024, 1024, 1024]
        self.max_positions = [2944, 3456, 3200, 3136]

    def setup_servos(self):
        """Setup servo communication"""
        if not DynamixelSDK_available:
            return
        # Implementation from original hardware_runner.py
        # ... (servo setup code)

    def start_camera_threads(self):
        """Start camera capture threads"""
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

    def _camera_capture_thread(self, camera_name, camera):
        """Camera capture thread"""
        while self.capture_active:
            try:
                ret, frame = camera.read()
                if ret:
                    try:
                        self.frame_queues[camera_name].put_nowait(frame)
                    except:
                        try:
                            self.frame_queues[camera_name].get_nowait()
                            self.frame_queues[camera_name].put_nowait(frame)
                        except Empty:
                            pass
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Camera {camera_name} error: {e}")
                break

    def stop_camera_threads(self):
        """Stop camera capture threads"""
        self.capture_active = False
        for camera_name, thread in self.camera_threads.items():
            thread.join(timeout=1.0)
        self.camera_threads.clear()

    def get_latest_frames(self):
        """Get latest frames from cameras"""
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

    def extract_hand_features(self, frame, camera_name):
        """Extract hand features from frame"""
        if frame is None:
            return [np.nan, np.nan], None

        try:
            if frame.size == 0 or len(frame.shape) != 3:
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

            return [np.nan, np.nan], display_frame

        except Exception as e:
            self.logger.error(f"Hand detection error in {camera_name}: {e}")
            return [np.nan, np.nan], frame.copy()

    def predict_joint_positions(self, features):
        """Predict joint positions using loaded model"""
        if self.model is None:
            return self.last_successful_positions.copy()

        try:
            features = np.array(features, dtype=np.float32).reshape(1, -1)

            if self.model_type == "xgb":
                predictions = self.model.predict(features)
                result = predictions[0]
                if not np.isfinite(result).all():
                    return self.last_successful_positions.copy()
                return result

            if not np.isfinite(features).all():
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
                return self.last_successful_positions.copy()
            return result

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.last_successful_positions.copy()

    def update_display(self, frames, left_features, right_features):
        """Update camera display with mode indicator"""
        if not self.show_display:
            return

        for camera_name, frame in frames.items():
            if frame is not None:
                display_frame = frame.copy()

                # Add mode indicator
                mode_text = f"MODE: {self.current_mode.upper()}"
                mode_color = (0, 255, 0) if self.current_mode == "camera" else (0, 255, 255)
                cv2.rectangle(display_frame, (10, 10), (200, 50), (0, 0, 0), -1)
                cv2.putText(display_frame, mode_text, (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                # Add hand tracking if in camera mode
                if self.current_mode == "camera":
                    features = left_features if camera_name == 'left' else right_features
                    x, y = features
                    if np.isfinite([x, y]).all():
                        color = (0, 255, 0) if camera_name == 'left' else (0, 255, 255)
                        cv2.circle(display_frame, (int(x), int(y)), 12, color, -1)
                        cv2.circle(display_frame, (int(x), int(y)), 15, (255, 255, 255), 2)

                window_name = f"{camera_name.title()} Camera"
                cv2.imshow(window_name, display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('v') and not self.mode_switching:
            self.switch_to_voice_mode()
        elif key == ord('c') and not self.mode_switching:
            self.switch_to_camera_mode()

    def switch_to_voice_mode(self):
        """Switch to voice control mode"""
        if self.current_mode != "voice":
            self.mode_switching = True
            self.current_mode = "voice"
            print("🎤 Switched to VOICE MODE - Say wake word to control Arduino")
            self.mode_switching = False

    def switch_to_camera_mode(self):
        """Switch to camera control mode"""
        if self.current_mode != "camera":
            self.mode_switching = True
            self.current_mode = "camera"
            print("📷 Switched to CAMERA MODE - Hand tracking controls robot")
            self.mode_switching = False

    def run_camera_mode_step(self):
        """Execute one step of camera mode"""
        frames = self.get_latest_frames()

        left_features, left_processed = self.extract_hand_features(frames.get('left'), 'left')
        right_features, right_processed = self.extract_hand_features(frames.get('right'), 'right')

        if left_processed is not None:
            frames['left'] = left_processed
        if right_processed is not None:
            frames['right'] = right_processed

        if self.show_display:
            self.update_display(frames, left_features, right_features)

        # Predict and control robot if model is available
        if self.model is not None:
            combined_features = left_features + right_features
            predicted_positions = self.predict_joint_positions(combined_features)

            # Apply safety and send to servos (implementation from original code)
            # ... (servo control code)

    def run_voice_mode_step(self):
        """Execute one step of voice mode"""
        # Get camera frames for display
        frames = self.get_latest_frames()
        left_features, right_features = [np.nan, np.nan], [np.nan, np.nan]

        if self.show_display:
            self.update_display(frames, left_features, right_features)

        # Voice recognition loop (simplified)
        try:
            wake_stream, wake_responses = start_stream(
                self.mic_index, self.mic_hint, self.debug, for_command=False
            )

            wake_start = time.time()
            got_wake = False

            for response in wake_responses:
                if time.time() - wake_start > 10.0:  # 10 second timeout
                    break

                # Check for wake word
                if response.results:
                    for res in response.results:
                        if res.alternatives and not res.is_final:
                            text = res.alternatives[0].transcript.strip()
                            if text and detect_wake_interim(text):
                                got_wake = True
                                break
                    if got_wake:
                        break

            wake_stream.__exit__(None, None, None)

            if got_wake:
                print("✅ Wake word detected - Say command")

                # Command recognition
                cmd_stream, cmd_responses = start_stream(
                    self.mic_index, self.mic_hint, self.debug, for_command=True
                )

                cmd_start = time.time()
                for response in cmd_responses:
                    if time.time() - cmd_start > 3.0:  # 3 second timeout
                        break

                    if response.results:
                        for res in response.results:
                            if res.alternatives and not res.is_final:
                                text = res.alternatives[0].transcript.strip()
                                if text:
                                    cmd = detect_cmd_interim(text)
                                    if cmd:
                                        self.handle_voice_command(cmd)
                                        break

                cmd_stream.__exit__(None, None, None)

        except Exception as e:
            self.logger.error(f"Voice recognition error: {e}")

    def handle_voice_command(self, command):
        """Handle recognized voice command"""
        print(f"🗣️ Command: {command}")

        if command == "EXIT":
            print("🛑 Exit command received")
            self.running = False
        elif command == "CAMERA_MODE":
            self.switch_to_camera_mode()
        elif command == "VOICE_MODE":
            self.switch_to_voice_mode()
        elif command in ["ON", "OFF"]:
            if self.arduino:
                send_arduino(self.arduino, command)

    def run_unified_loop(self):
        """Main unified control loop"""
        print(f"🤖 Starting unified system - Current mode: {self.current_mode}")

        # Start camera threads
        self.start_camera_threads()

        try:
            while self.running:
                loop_start = time.time()

                if self.current_mode == "camera":
                    self.run_camera_mode_step()
                elif self.current_mode == "voice":
                    self.run_voice_mode_step()

                # Frame rate control
                elapsed = time.time() - loop_start
                target_time = self.frame_time
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up resources...")

        # Stop camera threads
        self.stop_camera_threads()

        # Close cameras
        for camera in self.cameras.values():
            if camera is not None:
                try:
                    camera.release()
                except Exception as e:
                    self.logger.error(f"Failed to release camera: {e}")

        # Close display windows
        if self.show_display:
            cv2.destroyAllWindows()

        # Close Arduino
        if self.arduino and self.arduino.is_open:
            self.arduino.close()

        self.logger.info("Cleanup completed")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Voice + Hardware Runner")
    parser.add_argument("--model", help="Path to trained model (.joblib for XGBoost, .pth for PyTorch)")
    parser.add_argument("--config", default="hardware_config.json", help="Hardware configuration file")
    parser.add_argument("--arduino-port", default="/dev/arduino", help="Arduino port")
    parser.add_argument("--test", action='store_true', help="Run in test mode (no hardware control)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--display", action='store_true', help="Show camera windows")
    parser.add_argument("--mode", choices=["camera", "voice"], default="camera", help="Initial mode")
    parser.add_argument("--list-mics", action="store_true", help="List microphone devices")
    parser.add_argument("--mic-index", type=int, help="Microphone device index")
    parser.add_argument("--mic-hint", default=DEFAULT_MIC_HINT, help="Microphone device name hint")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")

    args = parser.parse_args()

    if args.list_mics:
        list_input_devices()
        return

    print("🤖 Unified Voice + Hardware Runner")
    print("=" * 50)
    print("Controls:")
    print("  'c' - Switch to camera mode")
    print("  'v' - Switch to voice mode")
    print("  'q' - Quit")
    print("Voice commands: 하이봇 → [켜/꺼/카메라/음성/종료]")
    print("=" * 50)

    try:
        runner = UnifiedVoiceHardwareRunner(
            model_path=args.model,
            hardware_config_path=args.config,
            arduino_port=args.arduino_port,
            test_mode=args.test,
            target_fps=args.fps,
            show_display=args.display,
            mic_index=args.mic_index,
            mic_hint=args.mic_hint,
            debug=args.debug
        )

        # Set initial mode
        runner.current_mode = args.mode

        runner.run_unified_loop()

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()