#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent Voice + Hardware Runner
Runs both systems simultaneously:
- Camera system: Real-time hand tracking → robot arm control (continuous)
- Voice system: Speech recognition → Arduino LED control (background thread)
- Both systems operate concurrently without interference
"""

# MARK: - Imports & Dependencies

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import json
import argparse
import logging
import mediapipe as mp
import os
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import joblib
import re
import unicodedata
import queue
import audioop
import ctypes
import ctypes.util
import struct
from typing import Optional, List, Tuple, Sequence, Callable

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

# MARK: - Configuration Constants

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
    # Arduino LED control
    "LED_OFF": ["꺼", "꺼줘", "불꺼", "불 꺼", "라이트오프", "라이트 오프", "끄자"],
    "LED_ON": ["켜", "켜줘", "불켜", "불 켜", "라이트온", "라이트 온", "키자"],
    "LED_BRIGHTER": ["밝게", "밝게해", "밝기 업", "더 밝게", "브라이트업"],
    "LED_DIMMER": ["어둡게", "어둡게해", "밝기 다운", "더 어둡게", "브라이트다운"],
    "LED_RED": ["빨간불", "빨간색", "빨강", "레드"],
    "LED_GREEN": ["초록불", "초록색", "초록", "그린"],
    "LED_BLUE": ["파란불", "파란색", "파랑", "블루"],
    "LED_YELLOW": ["노란불", "노란색", "노랑", "옐로우"],
    "LED_WHITE": ["하얀불", "하얀색", "흰색", "화이트"],
    "LED_RAINBOW": ["무지개", "무지개 불", "레인보우"],

    # Hand tracking control
    "TRACKING_ON": ["추적 시작", "시작", "추적 켜", "추적 온", "핸드 트래킹 켜", "손 추적 시작", "트래킹 시작", "트래킹 켜"],
    "GO_HOME": ["고 홈", "홈", "홈으로", "집으로", "원위치", "제자리"],

    # Robot position control
    "STOP": ["스톱", "정지", "멈춰", "멈춰줘", "멈춰라"],
    "EMERGENCY_RESET": ["리셋", "복구", "재시작", "긴급복구", "리셋해줘"],

    "EXIT": ["그만", "종료"]
}

LED_COMMAND_MAP = {
    "LED_ON": "ON",
    "LED_OFF": "OFF",
    "LED_BRIGHTER": "UP",
    "LED_DIMMER": "DOWN",
    "LED_RED": "RED",
    "LED_GREEN": "GREEN",
    "LED_BLUE": "BLUE",
    "LED_YELLOW": "YELLOW",
    "LED_WHITE": "WHITE",
    "LED_RAINBOW": "RAINBOW",
}

# Robot position constants
DEFAULT_HOME_POSITION = [2048, 3328, 1140, 3072]

# MARK: - PyTorch Model Classes

class BaseModel(nn.Module):
    """Base class with common weight initialization"""
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class SimpleTransformer(BaseModel):
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

    def forward(self, x):
        return self.network(x)


class ConfigurableFeedforward(BaseModel):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(BaseModel):
    """Residual block with feedforward layers"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def forward(self, x):
        residual = self.proj(x)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = out + residual
        out = self.norm(out)
        return out


class ResFeedforward(BaseModel):
    """Residual feedforward network"""
    def __init__(self, input_dim=4, output_dim=4, dropout=0.0):
        super().__init__()
        self.block_a = ResidualBlock(input_dim, 8, 16, dropout)
        self.block_b = ResidualBlock(16, 8, output_dim, dropout)
        self.long_skip = nn.Linear(input_dim, output_dim)
        self._init_weights()

    def forward(self, x):
        residual = self.long_skip(x)
        out = self.block_a(x)
        out = self.block_b(out)
        out = out + residual
        return out

# MARK: - Voice Recognition Utilities

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
    """Detect command in interim results (priority order)"""
    # Special case: check longer phrases first for TRACKING_ON
    if quick_contains(text, ["추적 시작", "트래킹 시작"]):
        return "TRACKING_ON"

    # Check commands in priority order
    priority_commands = [
        "EXIT", "EMERGENCY_RESET", "STOP", "GO_HOME", "TRACKING_ON",
        "LED_OFF", "LED_ON", "LED_BRIGHTER", "LED_DIMMER",
        "LED_RED", "LED_GREEN", "LED_BLUE", "LED_YELLOW", "LED_WHITE", "LED_RAINBOW"
    ]

    for cmd in priority_commands:
        if quick_contains(text, COMMAND_SYNONYMS[cmd]):
            return cmd

    return None

# MARK: - Microphone Stream

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
            yield pcm16k

# MARK: - Google Speech Client

def build_client_and_config(single_utter: bool):
    client = speech.SpeechClient()
    phrases = list(set(
        [WAKE_CANONICAL] + WAKE_VARIANTS +
        ["불 꺼 줘", "불켜 줘", "라이트 오프", "라이트 온",
         "라이트오프", "라이트온", "종료해", "끝내", "그만"] +
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

# MARK: - Main System Class

class ConcurrentVoiceHardwareRunner:
    # MARK: - Initialization
    def __init__(self, model_path=None, hardware_config_path='hardware_config.json',
                 arduino_port="/dev/arduino", test_mode=False, target_fps=60.0,
                 show_display=False, mic_index=None, mic_hint="blue", debug=False):

        # System control flags
        self.hand_tracking_enabled = False  # Hand tracking on/off
        self.voice_active = True           # Voice recognition system
        self.running = True
        self.robot_stopped = False         # Robot frozen at current position

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
        self.voice_thread = None
        self.arduino_reader_thread = None
        self.arduino_reader_active = False
        self._arduino_lock = threading.Lock()
        self._last_arduino_attempt = 0.0
        self._arduino_retry_interval = 5.0
        self._last_arduino_ping = 0.0
        self._arduino_keepalive_interval = 3.0

        # Thread synchronization locks
        self._frame_queue_lock = threading.RLock()  # For frame_queues dictionary access
        self._robot_state_lock = threading.RLock()  # For robot state flags
        self._mediapipe_lock = threading.RLock()    # For MediaPipe model calls 

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
        self.capture_active = threading.Event()  # Thread-safe flag for camera capture
        self.cameras = {}

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
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_successful_positions = DEFAULT_HOME_POSITION.copy()
        self.position_smoothing_alpha = 0.3


        self.last_positions = DEFAULT_HOME_POSITION.copy()
        self.emergency_stop = False
        self.safe_zone_min = [1280, 1920, 1120, 1664]
        self.safe_zone_max = [2944, 3456, 3200, 3136]

        # Safe holding position when tracking is disabled
        self.safe_holding_position = DEFAULT_HOME_POSITION.copy()

        # Track last sent positions to avoid duplicate commands
        self.last_sent_positions = None

        # Hand coordinate cache (per camera) - TTL: 0.5s
        self.cache_ttl = 0.0
        self.coord_cache = {
            'left':  {'xy': [np.nan, np.nan], 't': 0.0},
            'right': {'xy': [np.nan, np.nan], 't': 0.0},
        }

        # ThreadPoolExecutor for parallel hand tracking
        self.hand_tracking_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hand_track")

        # Initialize Arduino
        self.ensure_arduino_connected(force=True)

    # MARK: - Arduino Management

    def ensure_arduino_connected(self, force: bool = False) -> bool:
        """Ensure the Arduino serial connection is open, with optional forced retry."""
        if self.test_mode:
            return False

        with self._arduino_lock:
            if self.arduino and getattr(self.arduino, "is_open", False):
                return True

            now = time.time()
            if not force and (now - self._last_arduino_attempt) < self._arduino_retry_interval:
                return False

            self._last_arduino_attempt = now

            if self.arduino:
                try:
                    self.arduino.close()
                except Exception:
                    pass
                self.arduino = None

        # Open Arduino connection (integrated from open_arduino function)
        candidate = None
        try:
            ser = serial.Serial(self.arduino_port, 9600, timeout=1, write_timeout=1)
            time.sleep(2)
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass  # Some platforms don't support buffer reset
            self.logger.info(f"✓ Arduino connected ({self.arduino_port})")
            candidate = ser
        except Exception as e:
            self.logger.error(f"❌ Arduino connection failed: {e}")
            candidate = None

        with self._arduino_lock:
            if candidate and getattr(candidate, "is_open", False):
                self.arduino = candidate
                self._last_arduino_ping = time.time()
                return True

            self.arduino = None
            self._last_arduino_ping = 0.0
            return False

    def _mark_arduino_disconnected(self) -> None:
        """Close and clear Arduino connection state."""
        with self._arduino_lock:
            if self.arduino:
                try:
                    self.arduino.close()
                except Exception:
                    pass
            self.arduino = None
            self._last_arduino_attempt = 0.0
            self._last_arduino_ping = 0.0

    def send_arduino_command(self, cmd: str, quiet: bool = False) -> bool:
        """Send command to Arduino (integrated method)"""
        with self._arduino_lock:
            if not self.arduino or not getattr(self.arduino, "is_open", False):
                if not quiet:
                    self.logger.info(f"❌ Arduino not connected")
                return False
            try:
                self.arduino.write((cmd + "\n").encode())
                self.arduino.flush()
                if not quiet:
                    self.logger.info(f"👉 Arduino: {cmd}")
                return True
            except Exception as e:
                if not quiet:
                    self.logger.error(f"❌ Arduino send failed: {e}")
                return False

    # MARK: - Cache & Helper Methods

    def _cache_and_fill(self, camera_name: str, xy: List[float]) -> List[float]:
        """
        xy에 유효값이 있으면 캐시에 저장하고 그대로 반환.
        NaN이면 캐시에 든 값을 (TTL 이내라면) 반환, 없으면 [nan, nan].
        """
        now = time.time()
        cache = self.coord_cache.get(camera_name)
        if cache is None:
            cache = {'xy': [np.nan, np.nan], 't': 0.0}
            self.coord_cache[camera_name] = cache

        arr = np.asarray(xy, dtype=np.float32)
        if np.isfinite(arr).all():
            cache['xy'] = [float(arr[0]), float(arr[1])]
            cache['t'] = now
            return cache['xy']

        # 결측: 캐시 사용 (TTL==0 이면 무제한)
        age = now - cache['t']
        if np.isfinite(cache['xy']).all() and (self.cache_ttl <= 0.0 or age <= self.cache_ttl):
            return cache['xy']
        return [np.nan, np.nan]

    # MARK: - Model Loading

    def load_model(self):
        """Load XGBoost or PyTorch model (same as hardware_runner.py)"""
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
                    # Legacy 2-layer feedforward
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

            self.scaler_X = checkpoint.get('scaler_X', None)
            self.scaler_y = checkpoint.get('scaler_y', None)
            self.normalize = checkpoint.get('normalize', False)

            self.model_type = "torch"
            self.logger.info(f"PyTorch model loaded from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    # MARK: - Hardware Configuration

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

    # MARK: - Camera Setup

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

    # MARK: - MediaPipe Setup

    def setup_mediapipe(self):
        """Setup MediaPipe hand tracking"""
        self.mp_hands = mp.solutions.hands
        self.hands_left = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.1,
            model_complexity=0
        )
        self.hands_right = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.1,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils

    # MARK: - Display Setup

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
        self.logger.info("Display windows initialized - press 'q' to quit")

    # MARK: - Servo Setup

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

        # Set Position P Gain for all motors from config file
        position_p_gain_addr = 84
        p_gain_values = follower_config.get('position_p_gains', [])

        if p_gain_values:
            if len(p_gain_values) != len(self.servo_ids):
                self.logger.warning(f"Position P Gain array length ({len(p_gain_values)}) doesn't match motor count ({len(self.servo_ids)})")

            for i, servo_id in enumerate(self.servo_ids):
                if i < len(p_gain_values):
                    p_gain_value = int(p_gain_values[i])
                    try:
                        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                            self.port_handler, servo_id, position_p_gain_addr, p_gain_value
                        )
                        if dxl_comm_result != COMM_SUCCESS:
                            self.logger.warning(f"Failed to set Position P Gain for motor {servo_id}: Communication error")
                        elif dxl_error != 0:
                            self.logger.warning(f"Failed to set Position P Gain for motor {servo_id}: Error code {dxl_error}")
                        else:
                            self.logger.info(f"Position P Gain set to {p_gain_value} for motor {servo_id} (address {position_p_gain_addr})")
                    except Exception as e:
                        self.logger.error(f"Exception while setting Position P Gain for motor {servo_id}: {e}")
        else:
            self.logger.info("No Position P Gain values configured - using motor defaults")

        self.enable_torque()
        self.logger.info(f"Servos initialized on {port}")

    def enable_torque(self):
        """Enable torque for all servos"""
        if not DynamixelSDK_available:
            return

        torque_enable_addr = 64

        for servo_id in self.servo_ids:
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, torque_enable_addr, 1
                )
                if dxl_comm_result != COMM_SUCCESS:
                    self.logger.error(f"Failed to enable torque for servo {servo_id}")
                elif dxl_error != 0:
                    self.logger.error(f"Servo {servo_id} error: {dxl_error}")
                else:
                    self.logger.info(f"Torque enabled for servo {servo_id}")
            except Exception as e:
                self.logger.error(f"Exception enabling torque for servo {servo_id}: {e}")

    def disable_torque(self):
        """Disable torque for all servos"""
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

    # MARK: - Servo Control

    def send_servo_commands(self, positions):
        """Send position commands to servos"""
        if self.test_mode or not DynamixelSDK_available:
            return True

        if self.emergency_stop:
            self.logger.warning("Emergency stop active - not sending commands")
            return False

        # Skip if positions haven't changed (within 1 unit tolerance)
        if self.last_sent_positions is not None:
            positions_changed = False
            for i, pos in enumerate(positions):
                if i >= len(self.last_sent_positions) or abs(pos - self.last_sent_positions[i]) > 1:
                    positions_changed = True
                    break
            if not positions_changed:
                return True

        try:
            # Note: Safety check is performed in clamp_positions() before calling this method
            # Clear and add parameters
            self.group_sync_write.clearParam()

            for i, servo_id in enumerate(self.servo_ids):
                if i < len(positions):
                    position = int(positions[i])
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
                self.last_sent_positions = positions.copy()  # Track for duplicate prevention
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.emergency_stop = True

            return success

        except Exception as e:
            self.logger.error(f"Servo command failed: {e}")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.emergency_stop = True
            return False

    def clamp_positions(self, positions):
        """Apply safety constraints and smooth position changes"""
        if self.emergency_stop:
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

        self.last_positions = safe_positions
        return safe_positions

    def move_to_position(self, positions):
        """Clamp positions and send to servos (helper method)"""
        clamped = self.clamp_positions(positions)
        return self.send_servo_commands(clamped)

    # MARK: - Camera Threading

    def start_camera_threads(self):
        """Start camera capture threads"""
        self.capture_active.set()  # Set event to start capture
        for camera_name, camera in self.cameras.items():
            if camera is not None:
                with self._frame_queue_lock:
                    self.frame_queues[camera_name] = Queue(maxsize=10)
                thread = threading.Thread(
                    target=self._camera_capture_thread,
                    args=(camera_name, camera),
                    daemon=True
                )
                thread.start()
                self.camera_threads[camera_name] = thread

    def _camera_capture_thread(self, camera_name, camera):
        """Camera capture thread"""
        while self.capture_active.is_set():
            try:
                ret, frame = camera.read()
                if ret:
                    with self._frame_queue_lock:
                        try:
                            self.frame_queues[camera_name].put_nowait(frame)
                        except queue.Full:
                            # Queue is full, drop oldest frame and add new one
                            try:
                                self.frame_queues[camera_name].get_nowait()
                                self.frame_queues[camera_name].put_nowait(frame)
                            except Empty:
                                pass
                        except KeyError as e:
                            self.logger.error(f"[{camera_name}] Queue KeyError: {e}")
                            break

                time.sleep(0.001)
            except Exception as e:
                self.logger.error(f"Camera {camera_name} error: {e}")
                break

    def stop_camera_threads(self):
        """Stop camera capture threads"""
        self.capture_active.clear()  # Clear event to stop capture
        for camera_name, thread in self.camera_threads.items():
            thread.join(timeout=1.0)
        self.camera_threads.clear()

    def get_latest_frames(self):
        """Get latest frames from cameras"""
        frames = {}
        for camera_name in self.cameras.keys():
            try:
                frame = None
                with self._frame_queue_lock:
                    while True:
                        try:
                            frame = self.frame_queues[camera_name].get_nowait()
                        except Empty:
                            break

                frames[camera_name] = frame
            except KeyError:
                frames[camera_name] = None
        return frames

    # MARK: - Hand Tracking

    def extract_hand_features(self, frame, camera_name):
        """Extract hand features from frame (with cache fallback)"""
        if frame is None:
            # 프레임 자체가 없을 때도 캐시로 대체
            xy = self._cache_and_fill(camera_name, [np.nan, np.nan])
            return xy, None

        try:
            if frame.size == 0 or len(frame.shape) != 3:
                xy = self._cache_and_fill(camera_name, [np.nan, np.nan])
                return xy, frame.copy()

            hands_processor = self.hands_left if camera_name == 'left' else self.hands_right
            frame.flags.writeable = False
            # Use cv2.cvtColor for faster BGR→RGB conversion (IPP/SIMD optimized: 2-3x faster)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe models are thread-safe (separate instances for left/right cameras)
            # Each instance has its own TFLite interpreter and memory, allowing parallel processing
            results = hands_processor.process(rgb_frame)

            frame.flags.writeable = True
            # Only copy frame if display is enabled, otherwise use reference
            display_frame = frame.copy() if self.show_display else frame

            raw_xy = [np.nan, np.nan]

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Landmark drawing removed - index finger circle drawn in update_display() instead
                # (21 landmarks + connections drawing removed for performance: ~5-8ms saved per camera)
                index_tip = hand_landmarks.landmark[8]
                if 0 <= index_tip.x <= 1 and 0 <= index_tip.y <= 1:
                    h, w = frame.shape[:2]
                    x = index_tip.x * w
                    y = index_tip.y * h
                    if 0 <= x <= w and 0 <= y <= h:
                        raw_xy = [float(x), float(y)]

            # 캐시 보정 적용
            xy = self._cache_and_fill(camera_name, raw_xy)
            return xy, display_frame

        except Exception as e:
            self.logger.error(f"Hand detection error in {camera_name}: {e}")
            xy = self._cache_and_fill(camera_name, [np.nan, np.nan])
            return xy, frame.copy() if self.show_display else frame

    # MARK: - Prediction

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

    # MARK: - Display Update

    def update_display(self, frames, left_features, right_features):
        """Update camera display with concurrent mode indicators"""
        if not self.show_display:
            return

        for camera_name, frame in frames.items():
            if frame is not None:
                # Use the frame directly (already processed in extract_hand_features)
                display_frame = frame

                # Simplified status bar - only essential info
                cv2.rectangle(display_frame, (10, 10), (200, 50), (0, 0, 0), -1)

                # Robot status (most important)
                if self.robot_stopped:
                    robot_status = "STOPPED"
                    robot_color = (0, 165, 255)  # Orange
                elif self.hand_tracking_enabled:
                    robot_status = "TRACKING"
                    robot_color = (0, 255, 0)  # Green
                else:
                    robot_status = "HOME"
                    robot_color = (255, 255, 0)  # Cyan
                cv2.putText(display_frame, robot_status,
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, robot_color, 2)

                # Emergency stop warning (only when active)
                if self.emergency_stop:
                    cv2.rectangle(display_frame, (5, 5), (635, 475), (0, 0, 255), 5)
                    cv2.putText(display_frame, "EMERGENCY!",
                               (200, 240), cv2.FONT_HERSHEY_BOLD, 1.5, (0, 0, 255), 4)

                # Hand tracking - single circle only
                if self.hand_tracking_enabled:
                    features = left_features if camera_name == 'left' else right_features
                    x, y = features
                    if np.isfinite([x, y]).all():
                        color = (0, 255, 0) if camera_name == 'left' else (0, 255, 255)
                        cv2.circle(display_frame, (int(x), int(y)), 10, color, -1)

                window_name = f"{camera_name.title()} Camera"
                cv2.imshow(window_name, display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False

    # MARK: - Voice Recognition

    def _process_speech_stream(
        self,
        description: str,
        timeout: float,
        handler: Callable[[object], bool],
        *,
        for_command: bool = False,
        timeout_message: Optional[str] = None,
    ) -> str:
        """Run a streaming recognition session and delegate responses to handler."""
        stream_ctx = None
        try:
            stream_ctx, responses = start_stream(
                self.mic_index, self.mic_hint, self.debug, for_command=for_command
            )
        except Exception as e:
            self.logger.error(f"Failed to start {description} stream: {e}")
            return "start_error"

        start_time = time.time()
        try:
            for response in responses:
                if not self.running or not self.voice_active:
                    return "stopped"

                if timeout and time.time() - start_time > timeout:
                    if timeout_message:
                        print(timeout_message)
                    return "timeout"

                if response.results and handler(response):
                    return "success"
            return "no_match"
        except Exception as e:
            self.logger.error(f"{description.title()} recognition error: {e}")
            return "error"
        finally:
            if stream_ctx:
                try:
                    stream_ctx.__exit__(None, None, None)
                except Exception as close_err:
                    self.logger.error(f"Error closing {description} stream: {close_err}")
                    
    def voice_recognition_thread(self):
        """Background thread for continuous voice recognition"""
        print("🎤 Voice recognition thread started")
        consecutive_errors = 0
        max_consecutive_errors = 5

        def wake_handler(response) -> bool:
            for res in response.results:
                if res.alternatives and not res.is_final:
                    text = res.alternatives[0].transcript.strip()
                    if text and detect_wake_interim(text):
                        # Arduino에 무지개 효과 전송 (웨이크워드 시각적 확인)
                        self.send_arduino_command("LED_EFFECT:3", quiet=True)
                        return True
            return False

        def command_handler(response) -> bool:
            for res in response.results:
                if res.alternatives:
                    text = res.alternatives[0].transcript.strip()
                    if text:
                        cmd = detect_cmd_interim(text)
                        if cmd:
                            self.handle_voice_command(cmd)
                            return True
            return False

        while self.running and self.voice_active:
            try:
                wake_status = self._process_speech_stream(
                    "wake",
                    10.0,
                    wake_handler,
                    for_command=False,
                )

                if wake_status == "start_error":
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("⚠️ Too many voice recognition errors, stopping voice thread")
                        self.voice_active = False
                        break
                    time.sleep(2.0)
                    continue

                if wake_status == "error":
                    consecutive_errors += 1
                    continue

                if wake_status == "stopped":
                    break

                if wake_status != "success":
                    continue

                print("✅ Wake word detected - Listening for command...")

                cmd_status = self._process_speech_stream(
                    "command",
                    3.0,
                    command_handler,
                    for_command=False,
                    timeout_message="⏰ Command timeout",
                )

                if cmd_status == "start_error":
                    consecutive_errors += 1
                    continue

                if cmd_status == "stopped":
                    break

                if cmd_status == "error":
                    consecutive_errors += 1

                consecutive_errors = 0

            except Exception as e:
                self.logger.error(f"Voice recognition error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("⚠️ Too many voice recognition errors, stopping voice thread")
                    self.voice_active = False
                    break
                time.sleep(2.0)

        print("🔇 Voice recognition thread stopped")

    # MARK: - Arduino Reader Thread

    def arduino_reader_thread_func(self):
        """Background thread to continuously read Arduino serial data"""
        print("📖 Arduino reader thread started")
        while self.arduino_reader_active and self.running:
            try:
                with self._arduino_lock:
                    if self.arduino and getattr(self.arduino, "is_open", False):
                        # Read any available data to prevent buffer overflow
                        if self.arduino.in_waiting > 0:
                            try:
                                line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                                if line and self.debug:
                                    print(f"[Arduino] {line}")
                            except Exception as e:
                                if self.debug:
                                    self.logger.error(f"Arduino read error: {e}")
                time.sleep(0.01)  # Small delay to prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Arduino reader thread error: {e}")
                time.sleep(1.0)
        print("📖 Arduino reader thread stopped")

    # MARK: - Command Handling

    def handle_voice_command(self, command):
        """Handle recognized voice command"""
        print(f"🗣️ Voice Command: {command}")

        if command == "EXIT":
            print("🛑 Exit command received")
            self.running = False

        # Emergency reset
        elif command == "EMERGENCY_RESET":
            print("🔄 EMERGENCY RESET - Recovering system...")
            with self._robot_state_lock:
                if self.emergency_stop:
                    # Move to safe home position
                    if not self.test_mode:
                        safe_home = self.safe_holding_position
                        # Temporarily allow movement by clearing emergency stop
                        temp_emergency = self.emergency_stop
                        self.emergency_stop = False
                        success = self.move_to_position(safe_home)
                        if success:
                            self.consecutive_failures = 0
                            self.robot_stopped = False
                            self.hand_tracking_enabled = False
                            print("✅ System RECOVERED - Emergency stop cleared, moved to HOME")
                        else:
                            self.emergency_stop = temp_emergency
                            print("❌ Recovery FAILED - System still in emergency stop")
                    else:
                        self.emergency_stop = False
                        self.consecutive_failures = 0
                        print("✅ System RECOVERED (test mode)")
                else:
                    print("ℹ️ System is not in emergency stop")

        # Robot position control
        elif command == "STOP":
            with self._robot_state_lock:
                self.robot_stopped = True
                self.hand_tracking_enabled = False
            self.send_arduino_command("LED_EFFECT:11", quiet=True)  # 빨간색 깜빡임
            print("⏹️ Robot STOPPED at current position - Hand tracking disabled")

        elif command == "GO_HOME":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = False
                self.consecutive_failures = 0  # Reset communication failure counter
            self.send_arduino_command("LED_EFFECT:10", quiet=True)  # 노란색 깜빡임
            print("🏠 Going HOME - Moving to safe position")
            # Move to safe home position immediately
            if not self.test_mode:
                self.move_to_position(self.safe_holding_position)

        # Hand tracking control
        elif command == "TRACKING_ON":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = True
            self.send_arduino_command("LED_EFFECT:9", quiet=True)  # 파란색 깜빡임
            print("📷 Hand tracking ENABLED - Robot will follow hand movements")

        # Arduino LED control
        elif command in LED_COMMAND_MAP:
            arduino_cmd = LED_COMMAND_MAP[command]
            if self.ensure_arduino_connected(force=True):
                success = self.send_arduino_command(arduino_cmd)
                if not success:
                    print(f"⚠️ Arduino 전송 실패 - 연결을 재시도합니다 ({arduino_cmd})")
                    self._mark_arduino_disconnected()
            else:
                print(f"⚠️ Arduino not connected, cannot send {arduino_cmd}")


    # MARK: - Main Loop

    def run_concurrent_loop(self):
        """Main control loop - camera + voice simultaneously"""
        print("🤖 Starting system:")
        print(f"   📷 Hand tracking: {'ENABLED' if self.hand_tracking_enabled else 'DISABLED'}")
        print(f"   🎤 Voice control: {'ENABLED' if self.voice_active else 'DISABLED'}")

        # Start camera threads
        self.start_camera_threads()

        # Start Arduino reader thread
        if not self.test_mode:
            self.arduino_reader_active = True
            self.arduino_reader_thread = threading.Thread(target=self.arduino_reader_thread_func, daemon=True)
            self.arduino_reader_thread.start()

        # Start voice recognition thread
        if self.voice_active:
            self.voice_thread = threading.Thread(target=self.voice_recognition_thread, daemon=True)
            self.voice_thread.start()

        try:
            while self.running:
                loop_start = time.time()

                self.ensure_arduino_connected()

                arduino_ref = None
                should_ping = False
                if not self.test_mode:
                    with self._arduino_lock:
                        if self.arduino and getattr(self.arduino, "is_open", False):
                            arduino_ref = self.arduino
                            if time.time() - self._last_arduino_ping >= self._arduino_keepalive_interval:
                                should_ping = True

                if should_ping and arduino_ref:
                    if self.send_arduino_command("STATUS", quiet=True):
                        with self._arduino_lock:
                            self._last_arduino_ping = time.time()
                    else:
                        self._mark_arduino_disconnected()

                frames = self.get_latest_frames()

                if self.hand_tracking_enabled or self.show_display:
                    left_future = self.hand_tracking_executor.submit(
                        self.extract_hand_features, frames.get('left'), 'left'
                    )
                    right_future = self.hand_tracking_executor.submit(
                        self.extract_hand_features, frames.get('right'), 'right'
                    )

                    left_features, left_processed = left_future.result()
                    right_features, right_processed = right_future.result()

                    if left_processed is not None:
                        frames['left'] = left_processed
                    if right_processed is not None:
                        frames['right'] = right_processed
                else:
                    left_features = [np.nan, np.nan]
                    right_features = [np.nan, np.nan]

                if self.show_display:
                    self.update_display(frames, left_features, right_features)

                # Read robot state with lock to avoid race conditions
                with self._robot_state_lock:
                    robot_stopped = self.robot_stopped
                    hand_tracking_enabled = self.hand_tracking_enabled

                if robot_stopped:
                    pass
                elif hand_tracking_enabled and self.model is not None:
                    combined_features = left_features + right_features
                    final_positions = self.predict_joint_positions(combined_features)
                else:
                    final_positions = self.safe_holding_position

                self.move_to_position(final_positions)

                # Frame rate control
                elapsed = time.time() - loop_start
                target_time = self.frame_time
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user")
        finally:
            self.cleanup()

    # MARK: - Cleanup

    def cleanup(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up resources...")
        self.running = False

        # Return servos to safe home position and disable torque
        if not self.test_mode and DynamixelSDK_available:
            self.logger.info("Moving servos to home position...")
            try:
                # Move to safe home position
                if self.move_to_position(DEFAULT_HOME_POSITION):
                    time.sleep(1.0)  # Allow time to reach position
                    self.logger.info("Servos moved to home position")
                else:
                    self.logger.warning("Failed to move servos to home position")

                # Disable torque for all servos
                self.disable_torque()
                self.logger.info("Servo torque disabled")

            except Exception as e:
                self.logger.error(f"Error during servo cleanup: {e}")

        # Stop Arduino reader thread
        if self.arduino_reader_thread and self.arduino_reader_thread.is_alive():
            self.arduino_reader_active = False
            self.arduino_reader_thread.join(timeout=2.0)

        # Stop voice thread
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2.0)

        # Shutdown hand tracking executor
        if hasattr(self, 'hand_tracking_executor'):
            self.hand_tracking_executor.shutdown(wait=True, cancel_futures=True)
            self.logger.info("Hand tracking executor shutdown")

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

        # Turn off Arduino LED and close connection
        with self._arduino_lock:
            arduino_ref = self.arduino if self.arduino and getattr(self.arduino, "is_open", False) else None

        if arduino_ref:
            try:
                self.logger.info("Turning off Arduino LED...")
                if self.send_arduino_command("OFF"):
                    time.sleep(0.5)  # Brief delay for command to process
                    self.logger.info("Arduino LED turned off")
            except Exception as e:
                self.logger.error(f"Failed to turn off Arduino LED: {e}")

        self._mark_arduino_disconnected()

        # Close servo port
        if not self.test_mode and DynamixelSDK_available and hasattr(self, 'port_handler'):
            try:
                self.port_handler.closePort()
                self.logger.info("Servo port closed")
            except Exception as e:
                self.logger.error(f"Error closing servo port: {e}")

        self.logger.info("Cleanup completed")

# MARK: - Entry Point

def main():
    parser = argparse.ArgumentParser(description="Concurrent Voice + Hardware Runner")
    parser.add_argument("--model", help="Path to trained model (.joblib for XGBoost, .pth for PyTorch)")
    parser.add_argument("--config", default="hardware_config.json", help="Hardware configuration file")
    parser.add_argument("--arduino-port", default="/dev/arduino", help="Arduino port")
    parser.add_argument("--test", action='store_true', help="Run in test mode (no hardware control)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS")
    parser.add_argument("--display", action='store_true', help="Show camera windows")
    parser.add_argument("--no-camera", action='store_true', help="Disable camera control")
    parser.add_argument("--no-voice", action='store_true', help="Disable voice control")
    parser.add_argument("--list-mics", action="store_true", help="List microphone devices")
    parser.add_argument("--mic-index", type=int, help="Microphone device index")
    parser.add_argument("--mic-hint", default=DEFAULT_MIC_HINT, help="Microphone device name hint")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")

    args = parser.parse_args()

    if args.list_mics:
        list_input_devices()
        return

    print("🤖 Hand Tracking + Voice Control System")
    print("=" * 50)
    print("Main function: Camera hand tracking → Robot arm control")
    print("Voice control: Enable/disable hand tracking + Arduino LEDs")
    print()
    print("Voice commands after '하이봇':")
    print("  Hand tracking: '추적 시작'")
    print("  Robot control: '정지' (stop at current position) / '홈' (go to home)")
    print("  Arduino LEDs: '켜' / '꺼'")
    print("  Brightness: '밝게' / '어둡게'")
    print("  Colors: '빨간불', '초록불', '파란불', '노란불', '하얀불', '무지개'")
    print("  System: '종료'")
    print()
    print("Default: Hand tracking DISABLED")
    print("Robot states: TRACKING → STOPPED → HOME")
    print("Keyboard: 'q' to quit")
    print("=" * 50)

    try:
        runner = ConcurrentVoiceHardwareRunner(
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

        # Set concurrent modes
        runner.camera_active = not args.no_camera
        runner.voice_active = not args.no_voice

        if not runner.camera_active and not runner.voice_active:
            print("❌ Both camera and voice disabled. Nothing to run.")
            return

        runner.run_concurrent_loop()

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
