#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent Voice + Hardware Runner (+ shake measurement, + CSV logging)
- 멀티프로세싱 MediaPipe 손 추적
- 음성 인식 → 명령
- 로봇팔 제어
- ✅ 로봇팔이 목표각에 들어온 순간 아두이노 IMU로 들어오는 a_rms를 window 동안 수집해서 떨림 측정
- ✅ 측정 결과를 shake_log.csv 에 한 줄씩 append
- ❌ 손→모터 반응시간 측정 부분은 제거
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
import signal
import sys
import ctypes.util
import struct
from typing import Optional, List, Sequence, Callable
import multiprocessing as mp_module
from multiprocessing import Queue as MPQueue, Value as MPValue, Process as MPProcess
from collections import deque   # ✅ 떨림 측정용
import csv                      # ✅ CSV 저장용
from datetime import datetime   # ✅ 시각 기록용

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
    "TRACK_LEFT_HAND": ["왼손", "왼손 추적", "왼손만", "레프트 핸드", "왼쪽 손"],
    "TRACK_RIGHT_HAND": ["오른손", "오른손 추적", "오른손만", "라이트 핸드", "오른쪽 손"],
    "GO_HOME": ["고 홈", "홈", "홈으로", "집으로", "원위치", "제자리"],
    "STOP": ["스톱", "정지", "멈춰", "멈춰줘", "멈춰라"],
    "EXIT": ["그만", "종료", "끝내"]
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

DEFAULT_HOME_POSITION = [2048, 3328, 1140, 1600, 2048]

# MARK: - Model classes

class BaseModel(nn.Module):
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class SimpleTransformer(BaseModel):
    def __init__(self, input_dim=4, output_dim=5, d_model=8, nhead=1,
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
    def __init__(self, input_dim=4, output_dim=5, hidden_sizes=None, dropout=0.0):
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
    def __init__(self, input_dim=4, output_dim=5, dropout=0.0):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 8)
        self.act = nn.ReLU()
        self.block_a = ResidualBlock(8, 16, 32, dropout)
        self.block_b = ResidualBlock(32, 16, 8, dropout)
        self.fc_out = nn.Linear(8, output_dim)
        self.long_skip = nn.Linear(input_dim, output_dim)
        self._init_weights()
    def forward(self, x):
        residual = self.long_skip(x)
        out = self.fc_in(x)
        out = self.act(out)
        out = self.block_a(out)
        out = self.block_b(out)
        out = self.fc_out(out)
        out = out + residual
        return out

# MARK: - Voice utilities

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

def quick_contains(text: str, keys: List[str]) -> bool:
    s = normalize(text)
    return any(normalize(k) in s for k in keys)

def detect_cmd_interim(text: str) -> Optional[str]:
    if quick_contains(text, ["추적 시작", "트래킹 시작"]):
        return "TRACK_LEFT_HAND"
    priority_commands = [
        "EXIT", "STOP", "GO_HOME",
        "TRACK_LEFT_HAND", "TRACK_RIGHT_HAND",
        "LED_OFF", "LED_ON", "LED_BRIGHTER", "LED_DIMMER",
        "LED_RED", "LED_GREEN", "LED_BLUE", "LED_YELLOW", "LED_WHITE", "LED_RAINBOW"
    ]
    for cmd in priority_commands:
        if cmd in COMMAND_SYNONYMS and quick_contains(text, COMMAND_SYNONYMS[cmd]):
            return cmd
    return None

# MARK: - Microphone stream

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
        self._debug = debug
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
                except (OSError, IOError) as e:
                    if self._debug:
                        print(f"  DEBUG: Failed to open mic with rate={r}, ch={ch}: {e}")
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
        if self._pa:
            self._pa.terminate()

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

# MARK: - Hand tracking worker

def hand_tracking_worker_process(camera_name, input_queue, result_queue, hand_filter_mode, processing_enabled, config):
    import logging, time as _time
    logger = logging.getLogger(f"worker.{camera_name}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s'))
    logger.addHandler(handler)

    mp_hands = mp.solutions.hands
    hands_processor = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=config['min_detection_confidence'],
        min_tracking_confidence=config['min_tracking_confidence'],
        model_complexity=config['model_complexity']
    )

    coord_cache = {'xy': [np.nan, np.nan], 't': 0.0}
    frame_counter = 0
    last_handedness = None
    cache_ttl = config['cache_ttl']
    process_every_n_frames = config['process_every_n_frames']

    smoothed_xy = None
    smooth_alpha = 1.0
    deadzone_px = 8.0
    jump_px = 60.0

    def cache_and_fill(xy):
        nonlocal coord_cache
        now = _time.perf_counter()
        arr = np.asarray(xy, dtype=np.float32)
        if np.isfinite(arr).all():
            coord_cache['xy'] = [float(arr[0]), float(arr[1])]
            coord_cache['t'] = now
            return coord_cache['xy']
        age = now - coord_cache['t']
        if np.isfinite(coord_cache['xy']).all() and (cache_ttl <= 0.0 or age <= cache_ttl):
            return coord_cache['xy']
        return [np.nan, np.nan]

    while processing_enabled.value:
        try:
            item = input_queue.get(timeout=0.1)
            if item is None:
                break

            if isinstance(item, tuple):
                frame, t_capture = item
            else:
                frame = item
                t_capture = _time.perf_counter()

            frame_counter += 1
            should_process = (frame_counter % process_every_n_frames == 0)

            raw_xy = [np.nan, np.nan]
            handedness = last_handedness

            if should_process:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands_processor.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        selected_hand_idx = None
                        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            current_handedness = None
                            if results.multi_handedness and i < len(results.multi_handedness):
                                hand_info = results.multi_handedness[i]
                                raw_handedness = hand_info.classification[0].label
                                if camera_name == 'left':
                                    current_handedness = "Left" if raw_handedness == "Right" else "Right"
                                else:
                                    current_handedness = "Right" if raw_handedness == "Left" else "Left"

                            filter_val = hand_filter_mode.value
                            filter_mode_str = None if filter_val == 0 else ("Left" if filter_val == 1 else "Right")

                            if filter_mode_str is not None:
                                if current_handedness == filter_mode_str:
                                    selected_hand_idx = i
                                    handedness = current_handedness
                                    break
                            else:
                                selected_hand_idx = i
                                handedness = current_handedness
                                break

                        if selected_hand_idx is not None:
                            hand_landmarks = results.multi_hand_landmarks[selected_hand_idx]
                            index_tip = hand_landmarks.landmark[8]
                            if 0 <= index_tip.x <= 1 and 0 <= index_tip.y <= 1:
                                h, w = frame.shape[:2]
                                x = index_tip.x * w
                                y = index_tip.y * h
                                if 0 <= x <= w and 0 <= y <= h:
                                    raw_xy = [float(x), float(y)]
                            last_handedness = handedness
                        else:
                            last_handedness = None
                    else:
                        last_handedness = None

                except Exception as e:
                    logger.error(f"MediaPipe processing error: {e}", exc_info=True)
                    try:
                        result_queue.put_nowait(([np.nan, np.nan], None, t_capture))
                    except queue.Full:
                        pass

            xy_final = cache_and_fill(raw_xy)

            if smoothed_xy is None or not np.isfinite(smoothed_xy).all():
                smoothed_xy = xy_final
            else:
                new_x, new_y = xy_final
                old_x, old_y = smoothed_xy
                if np.isfinite([new_x, new_y]).all():
                    dist = ((new_x - old_x)**2 + (new_y - old_y)**2) ** 0.5
                    if dist < deadzone_px:
                        pass
                    elif dist > jump_px:
                        smoothed_xy = [new_x, new_y]
                    else:
                        smoothed_xy = [
                            smooth_alpha * new_x + (1 - smooth_alpha) * old_x,
                            smooth_alpha * new_y + (1 - smooth_alpha) * old_y,
                        ]

            try:
                result_queue.put_nowait((smoothed_xy, last_handedness, t_capture))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                    result_queue.put_nowait((smoothed_xy, last_handedness, t_capture))
                except (queue.Full, queue.Empty):
                    pass

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)

    hands_processor.close()
    logger.info(f"Hand tracking worker shutdown for {camera_name}")

# MARK: - Main Runner

class ConcurrentVoiceHardwareRunner:
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_POSITION = 132

    def __init__(self, model_path=None, hardware_config_path='hardware_config.json',
                 arduino_port="/dev/arduino", test_mode=False, target_fps=60.0,
                 show_display=False, mic_index=None, mic_hint="blue", debug=False):

        self._last_frames = {}
        self.hand_tracking_enabled = False
        self.voice_active = True
        self.running = True
        self.robot_stopped = False

        self.mediapipe_process_every_n_frames = 1

        self.model_path = model_path
        self.test_mode = test_mode
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.show_display = show_display

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

        self._frame_queue_lock = threading.RLock()
        self._robot_state_lock = threading.RLock()
        self._mediapipe_lock = threading.RLock()

        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self.model = None
        self.model_type = None
        self.scaler_X = None
        self.scaler_y = None
        self.normalize = False

        if model_path:
            self.load_model()

        self.load_hardware_config(hardware_config_path)

        self.frame_queues = {}
        self.camera_threads = {}
        self.capture_active = threading.Event()
        self.cameras = {}

        self.setup_cameras()
        self.setup_mediapipe()
        if self.show_display:
            self.setup_display_windows()

        if not test_mode and DynamixelSDK_available:
            self.setup_servos()
        else:
            self.logger.info("🧪 TEST MODE - Hardware control disabled")
            self.setup_servo_defaults()

        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_successful_positions = DEFAULT_HOME_POSITION.copy()
        self.position_smoothing_alpha = 0.1
        self.last_positions = DEFAULT_HOME_POSITION.copy()
        self.emergency_stop = False
        self.safe_zone_min = [1024, 1900, 1024, 1024, 512]
        self.safe_zone_max = [2944, 3520, 3340, 3136, 4096]
        self.safe_holding_position = DEFAULT_HOME_POSITION.copy()
        self.last_sent_positions = None

        self.prev_left_xy = None
        self.prev_right_xy = None
        self.input_smooth_alpha = 1.0
        self.input_deadzone_px = 2.0

        self.cache_ttl = 0.0
        self.main_coord_cache = {
            'left': {'xy': [np.nan, np.nan], 't': 0.0},
            'right': {'xy': [np.nan, np.nan], 't': 0.0},
        }

        self.mp_input_queue_left = MPQueue(maxsize=2)
        self.mp_input_queue_right = MPQueue(maxsize=2)
        self.mp_result_queue_left = MPQueue(maxsize=2)
        self.mp_result_queue_right = MPQueue(maxsize=2)

        self.queue_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="queue_processor")

        self.mp_hand_filter_mode = MPValue('i', 0)
        self.mp_processing_enabled = MPValue('b', True)
        self.mp_workers = {}

        self.mp_config = {
            'min_detection_confidence': 0.4,
            'min_tracking_confidence': 0.4,
            'model_complexity': 1,
            'process_every_n_frames': self.mediapipe_process_every_n_frames,
            'cache_ttl': self.cache_ttl,
            'show_display': self.show_display
        }

        self.ensure_arduino_connected(force=True)

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # ===== Shake measurement state =====
        self.accel_queue = deque(maxlen=512)
        self.accel_lock = threading.Lock()
        self.shake_measure_enabled = True
        self.shake_tol_ticks = 10
        self.shake_window_s = 2.0
        self.shake_servo_index = 0
        self.shake_cooldown_s = 0.5
        self._shake_active = False
        self._shake_start_t = 0.0
        self._shake_last_ts = 0.0
        self._shake_cooldown_until = 0.0
        self._prev_within_tol = False
        self._shake_max_a = 0.0
        self._shake_min_a = float('inf')
        self._shake_sum_a = 0.0
        self._shake_count = 0

        # ===== CSV logging =====
        self.shake_csv_path = "shake_log.csv"
        self._ensure_shake_csv_header()

    # MARK: - CSV helper
    def _ensure_shake_csv_header(self):
        """CSV 파일이 없으면 헤더를 만들어 둔다."""
        if not os.path.exists(self.shake_csv_path):
            try:
                with open(self.shake_csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "end_time",
                        "window_sec",
                        "avg_a_g",
                        "min_a_g",
                        "max_a_g",
                        "samples",
                        "servo_index",
                        "tol_ticks",
                    ])
            except Exception as e:
                self.logger.error(f"Failed to create CSV file: {e}")

    def _append_shake_csv(self, window_sec: float, avg_a: float, min_a: float,
                          max_a: float, samples: int, servo_idx: int, tol_ticks: int):
        """측정 결과 한 줄을 CSV에 append"""
        try:
            with open(self.shake_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    f"{window_sec:.3f}",
                    f"{avg_a:.6f}",
                    f"{min_a:.6f}",
                    f"{max_a:.6f}",
                    samples,
                    servo_idx,
                    tol_ticks,
                ])
        except Exception as e:
            self.logger.error(f"Failed to write to CSV: {e}")

    # MARK: - Signal

    def _signal_handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        self.logger.info(f"Received signal {sig_name}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)

    # MARK: - Arduino

    def ensure_arduino_connected(self, force: bool = False) -> bool:
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
                try: self.arduino.close()
                except Exception: pass
                self.arduino = None
        try:
            ser = serial.Serial(self.arduino_port, 9600, timeout=1, write_timeout=1)
            time.sleep(2)
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass
            self.arduino = ser
            self._last_arduino_ping = time.time()
            self.logger.info(f"✓ Arduino connected ({self.arduino_port}) @9600")
            return True
        except Exception as e:
            self.logger.error(f"❌ Arduino connection failed: {e}")
            self.arduino = None
            self._last_arduino_ping = 0.0
            return False

    def _mark_arduino_disconnected(self) -> None:
        with self._arduino_lock:
            if self.arduino:
                try: self.arduino.close()
                except Exception: pass
            self.arduino = None
            self._last_arduino_attempt = 0.0
            self._last_arduino_ping = 0.0

    def send_arduino_command(self, cmd: str, quiet: bool = False) -> bool:
        with self._arduino_lock:
            if not self.arduino or not getattr(self.arduino, "is_open", False):
                if not quiet:
                    self.logger.info("❌ Arduino not connected")
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

    # MARK: - Model loading

    def load_model(self):
        self.model_type = None
        self.normalize = False
        self.scaler_X = None
        self.scaler_y = None
        if self.model_path.lower().endswith(".joblib"):
            self.logger.info(f"Loading XGBoost model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_type = "xgb"
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
            if 'feedforward' in arch_lower and 'res' not in arch_lower:
                hidden_sizes = config.get('hidden_sizes')
                has_layer_norm = any('.1.' in key for key in state_dict.keys())
                if not hidden_sizes and not has_layer_norm:
                    self.model = SimpleTransformer(
                        input_dim=4,
                        output_dim=5,
                        d_model=config.get('d_model', config.get('dim_feedforward', 12)),
                        nhead=config.get('nhead', 1),
                        num_layers=1,
                        dim_feedforward=config.get('dim_feedforward', 12),
                        dropout=0.0
                    ).to(self.device)
                else:
                    if not hidden_sizes:
                        num_layers = max(int(config.get('num_layers', 1)), 1)
                        dim_size = int(config.get('dim_feedforward', 12))
                        hidden_sizes = tuple(dim_size for _ in range(num_layers))
                    else:
                        hidden_sizes = tuple(int(size) for size in hidden_sizes)
                    dropout = float(config.get('dropout', 0.0))
                    self.model = ConfigurableFeedforward(
                        input_dim=4, output_dim=5,
                        hidden_sizes=hidden_sizes,
                        dropout=dropout
                    ).to(self.device)
            elif 'res' in arch_lower:
                self.model = ResFeedforward(
                    input_dim=4,
                    output_dim=5,
                    dropout=config.get('dropout', 0.0)
                ).to(self.device)
            else:
                self.model = SimpleTransformer(
                    input_dim=4,
                    output_dim=5,
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

    # MARK: - Hardware config
    def load_hardware_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Hardware config loaded: {config_path}")
        except FileNotFoundError:
            self.logger.warning(f"Hardware config not found: {config_path}, using defaults")
            self.config = {}

    # MARK: - Cameras
    def setup_cameras(self):
        self.cameras = {}
        camera_config = self.config.get('cameras', {})
        left_config = camera_config.get('cam_left', {'id': '/dev/cam_left', 'enabled': True})
        if left_config.get('enabled', True):
            left_id = left_config.get('id', '/dev/cam_left')
            self.cameras['left'] = self._setup_single_camera('left', left_id, left_config)
        else:
            self.cameras['left'] = None
        right_config = camera_config.get('cam_right', {'id': '/dev/cam_right', 'enabled': True})
        if right_config.get('enabled', True):
            right_id = right_config.get('id', '/dev/cam_right')
            self.cameras['right'] = self._setup_single_camera('right', right_id, right_config)
        else:
            self.cameras['right'] = None

    def _setup_single_camera(self, camera_name, camera_id, config):
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
        self.mp_drawing = mp.solutions.drawing_utils

    def setup_display_windows(self):
        if self.cameras.get('left') is not None:
            cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Left Camera', 640, 480)
            cv2.moveWindow('Left Camera', 100, 100)
        if self.cameras.get('right') is not None:
            cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Right Camera', 640, 480)
            cv2.moveWindow('Right Camera', 780, 100)
        self.logger.info("Display windows initialized - press 'q' to quit")

    # MARK: - Servo setup
    def setup_servo_defaults(self):
        robot_config = self.config.get('robot_arms', {})
        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4, 5])
        self.min_positions = [1024, 1900, 1024, 1024, 512]
        self.max_positions = [2944, 3520, 3340, 3136, 4096]

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
        self.servo_ids = robot_config.get('motor_ids', [1, 2, 3, 4, 5])
        self.min_positions = [1024, 1900, 1024, 1024, 512]
        self.max_positions = [2944, 3520, 3340, 3136, 4096]
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
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    self.logger.error(f"Torque enable failed for servo {servo_id}")
            except Exception as e:
                self.logger.error(f"Exception enabling torque for servo {servo_id}: {e}")

    def disable_torque(self):
        if not DynamixelSDK_available:
            return
        torque_enable_addr = 64
        for servo_id in self.servo_ids:
            try:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, torque_enable_addr, 0
                )
            except Exception as e:
                self.logger.error(f"Exception disabling torque for servo {servo_id}: {e}")

    def send_servo_commands(self, positions):
        if self.test_mode or not DynamixelSDK_available:
            return True
        if self.emergency_stop:
            self.logger.warning("Emergency stop active - not sending commands")
            return False
        if self.last_sent_positions is not None:
            changed = False
            for i, pos in enumerate(positions):
                if i >= len(self.last_sent_positions) or abs(pos - self.last_sent_positions[i]) > 1:
                    changed = True
                    break
            if not changed:
                return True
        try:
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
            dxl_comm_result = self.group_sync_write.txPacket()
            success = dxl_comm_result == COMM_SUCCESS
            if success:
                self.last_successful_positions = positions.copy()
                self.last_sent_positions = positions.copy()
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
        clamped = self.clamp_positions(positions)
        return self.send_servo_commands(clamped)

    # ==== Dynamixel read helpers for shake ====
    def _dxl_read_present_position(self, servo_id: int) -> Optional[int]:
        if self.test_mode or not DynamixelSDK_available or not hasattr(self, 'packet_handler'):
            return None
        try:
            pos, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
                self.port_handler, servo_id, self.ADDR_PRESENT_POSITION
            )
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                return int(pos)
        except Exception:
            pass
        return None

    def _dxl_read_all_present_positions(self) -> Optional[List[int]]:
        if self.test_mode or not DynamixelSDK_available or not hasattr(self, 'packet_handler'):
            return None
        poses = []
        for sid in self.servo_ids:
            p = self._dxl_read_present_position(sid)
            if p is None:
                return None
            poses.append(p)
        return poses

    # MARK: - Camera threading
    def start_camera_threads(self):
        self.capture_active.set()
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
        while self.capture_active.is_set():
            try:
                ret, frame = camera.read()
                if ret:
                    t_cap = time.perf_counter()
                    with self._frame_queue_lock:
                        try:
                            self.frame_queues[camera_name].put_nowait((frame, t_cap))
                        except queue.Full:
                            try:
                                self.frame_queues[camera_name].get_nowait()
                            except Empty:
                                pass
                            try:
                                self.frame_queues[camera_name].put_nowait((frame, t_cap))
                            except queue.Full:
                                pass
                time.sleep(0.001)
            except Exception as e:
                self.logger.error(f"Camera {camera_name} error: {e}")
                break

    def stop_camera_threads(self):
        self.capture_active.clear()
        for camera_name, thread in self.camera_threads.items():
            thread.join(timeout=1.0)
        self.camera_threads.clear()

    def get_latest_frames(self):
        frames = {}
        for camera_name in self.cameras.keys():
            try:
                with self._frame_queue_lock:
                    try:
                        item = self.frame_queues[camera_name].get_nowait()
                        self._last_frames[camera_name] = item
                    except Empty:
                        item = self._last_frames.get(camera_name, None)
                frames[camera_name] = item
            except KeyError:
                frames[camera_name] = None
        return frames

    # MARK: - Queue processing
    def _process_camera_queue(self, camera_name, frame_item, input_queue, result_queue):
        if frame_item is not None:
            try:
                input_queue.put_nowait(frame_item)
            except queue.Full:
                pass

        features = [np.nan, np.nan]
        handedness = None
        t_capture = None
        try:
            result = result_queue.get(timeout=0.001)
            features, handedness, t_capture = result
            if np.isfinite(features).all():
                self.main_coord_cache[camera_name]['xy'] = features
                self.main_coord_cache[camera_name]['t'] = time.time()
        except queue.Empty:
            cached = self.main_coord_cache[camera_name]
            if self.cache_ttl <= 0.0 or (time.time() - cached['t']) <= self.cache_ttl:
                features = cached['xy']
        return features, handedness, t_capture

    # MARK: - Input smoothing
    def _smooth_xy(self, xy, is_left=True):
        arr = np.array(xy, dtype=np.float32)
        prev = self.prev_left_xy if is_left else self.prev_right_xy
        if not np.isfinite(arr).all():
            if prev is not None and np.isfinite(prev).all():
                return prev.tolist() if isinstance(prev, np.ndarray) else prev
            return xy
        if prev is None or not np.isfinite(prev).all():
            new_xy = arr
        else:
            dist = float(np.linalg.norm(arr - prev))
            if dist < self.input_deadzone_px:
                new_xy = prev
            else:
                alpha = self.input_smooth_alpha
                new_xy = alpha * arr + (1 - alpha) * prev
        if is_left:
            self.prev_left_xy = new_xy
        else:
            self.prev_right_xy = new_xy
        return new_xy.tolist()

    # MARK: - Display
    def update_display(self, frames, left_features, right_features, left_handedness=None, right_handedness=None):
        if not self.show_display:
            return
        for camera_name, item in frames.items():
            if item is not None:
                frame, _ = item
                display_frame = frame
                cv2.rectangle(display_frame, (10, 10), (200, 50), (0, 0, 0), -1)
                if self.robot_stopped:
                    robot_status = "STOPPED"
                    robot_color = (0, 165, 255)
                elif self.hand_tracking_enabled:
                    robot_status = "TRACKING"
                    robot_color = (0, 255, 0)
                else:
                    robot_status = "HOME"
                    robot_color = (255, 255, 0)
                cv2.putText(display_frame, robot_status,
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, robot_color, 2)
                if self.hand_tracking_enabled:
                    features = left_features if camera_name == 'left' else right_features
                    x, y = features
                    if np.isfinite([x, y]).all():
                        color = (0, 255, 0) if camera_name == 'left' else (0, 255, 255)
                        cv2.circle(display_frame, (int(x), int(y)), 10, color, -1)
                window_name = f"{camera_name.title()} Camera"
                cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False

    # MARK: - Prediction
    def predict_joint_positions(self, features):
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

    # MARK: - Voice thread
    def _process_speech_stream(self, description, timeout, handler, *, for_command=False, timeout_message=None):
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
        print("🎤 Voice recognition thread started")
        consecutive_errors = 0
        max_consecutive_errors = 5

        def wake_handler(response) -> bool:
            for res in response.results:
                if res.alternatives and not res.is_final:
                    text = res.alternatives[0].transcript.strip()
                    if text and (is_wake_word(text) or quick_contains(text, [WAKE_CANONICAL])):
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
                ws = self._process_speech_stream("wake", 10.0, wake_handler)
                if ws != "success":
                    continue
                print("✅ Wake word detected - Listening for command...")
                cs = self._process_speech_stream("command", 3.0, command_handler, timeout_message="⏰ Command timeout")
                if cs == "stopped":
                    break
            except Exception as e:
                self.logger.error(f"Voice recognition error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("⚠️ Too many voice recognition errors, stopping voice thread")
                    self.voice_active = False
                    break
        print("🔇 Voice recognition thread stopped")

    # MARK: - Arduino reader (CSV 파싱 포함)
    def arduino_reader_thread_func(self):
        print("📖 Arduino reader thread started")
        while self.arduino_reader_active and self.running:
            try:
                with self._arduino_lock:
                    if self.arduino and getattr(self.arduino, "is_open", False):
                        ser = self.arduino
                    else:
                        ser = None
                if ser and ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    parts = line.split(',')
                    if len(parts) >= 3:
                        pc_ts = time.perf_counter()
                        try:
                            g_rms = float(parts[1])
                            a_rms = float(parts[2])
                            with self.accel_lock:
                                self.accel_queue.append((pc_ts, a_rms, g_rms))
                        except ValueError:
                            if self.debug and line:
                                print(f"[Arduino raw] {line}")
                    else:
                        if self.debug and line:
                            print(f"[Arduino] {line}")
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Arduino reader thread error: {e}")
                time.sleep(1.0)
        print("📖 Arduino reader thread stopped")

    # MARK: - Voice commands
    def handle_voice_command(self, command):
        print(f"🗣️ Voice Command: {command}")
        if command == "EXIT":
            self.running = False
        elif command == "STOP":
            with self._robot_state_lock:
                self.robot_stopped = True
                self.hand_tracking_enabled = False
            self.send_arduino_command("LED_EFFECT:11", quiet=True)
        elif command == "GO_HOME":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = False
                self.consecutive_failures = 0
            self.send_arduino_command("LED_EFFECT:10", quiet=True)
            if not self.test_mode:
                self.move_to_position(self.safe_holding_position)
        elif command == "TRACK_LEFT_HAND":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = True
            self.mp_hand_filter_mode.value = 1
        elif command == "TRACK_RIGHT_HAND":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = True
            self.mp_hand_filter_mode.value = 2
        elif command in LED_COMMAND_MAP:
            arduino_cmd = LED_COMMAND_MAP[command]
            if self.ensure_arduino_connected(force=True):
                self.send_arduino_command(arduino_cmd)

    # MARK: - Shake measurement helpers
    def _start_shake_window(self, now_perf: float):
        self._shake_active = True
        self._shake_start_t = now_perf
        self._shake_last_ts = now_perf
        self._shake_max_a = 0.0
        self._shake_min_a = float('inf')
        self._shake_sum_a = 0.0
        self._shake_count = 0
        print(f"📈 Shake measurement START ({self.shake_window_s:.1f}s) — tol ±{self.shake_tol_ticks} ticks")

    def _finish_shake_window(self, now_perf: float):
        if self._shake_count == 0:
            print("🧪 Shake result: no accelerometer samples (check Arduino stream)")
        else:
            avg_a = self._shake_sum_a / self._shake_count
            print(
                f"🧪 Shake result over {self.shake_window_s:.1f}s -> "
                f"avg={avg_a:.4f} g, min={self._shake_min_a:.4f} g, max={self._shake_max_a:.4f} g "
                f"(samples={self._shake_count})"
            )
            # ✅ CSV로도 남긴다
            self._append_shake_csv(
                self.shake_window_s,
                avg_a,
                self._shake_min_a,
                self._shake_max_a,
                self._shake_count,
                self.shake_servo_index,
                self.shake_tol_ticks
            )
        self._shake_active = False
        self._shake_cooldown_until = now_perf + self.shake_cooldown_s

    # MARK: - Main loop
    def run_concurrent_loop(self):
        print("🤖 Starting system:")
        print(f"   📷 Hand tracking: {'ENABLED' if self.hand_tracking_enabled else 'DISABLED'}")
        print(f"   🎤 Voice control: {'ENABLED' if self.voice_active else 'DISABLED'}")

        self.start_camera_threads()
        self.start_hand_tracking_processes()

        if not self.test_mode:
            self.arduino_reader_active = True
            self.arduino_reader_thread = threading.Thread(target=self.arduino_reader_thread_func, daemon=True)
            self.arduino_reader_thread.start()

        if self.voice_active:
            self.voice_thread = threading.Thread(target=self.voice_recognition_thread, daemon=True)
            self.voice_thread.start()

        try:
            while self.running:
                loop_start = time.time()
                self.ensure_arduino_connected()

                frames = self.get_latest_frames()

                if self.hand_tracking_enabled or self.show_display:
                    left_future = self.queue_executor.submit(
                        self._process_camera_queue,
                        'left',
                        frames.get('left'),
                        self.mp_input_queue_left,
                        self.mp_result_queue_left
                    )
                    right_future = self.queue_executor.submit(
                        self._process_camera_queue,
                        'right',
                        frames.get('right'),
                        self.mp_input_queue_right,
                        self.mp_result_queue_right
                    )
                    left_features, left_handedness, left_tcap = left_future.result()
                    right_features, right_handedness, right_tcap = right_future.result()

                    left_features = self._smooth_xy(left_features, is_left=True)
                    right_features = self._smooth_xy(right_features, is_left=False)
                else:
                    left_features = [np.nan, np.nan]
                    right_features = [np.nan, np.nan]
                    left_tcap = right_tcap = None

                if self.show_display:
                    self.update_display(frames, left_features, right_features)

                with self._robot_state_lock:
                    robot_stopped = self.robot_stopped
                    hand_tracking_enabled = self.hand_tracking_enabled

                if robot_stopped:
                    final_positions = self.last_positions
                elif hand_tracking_enabled and self.model is not None:
                    combined_features = left_features + right_features
                    final_positions = self.predict_joint_positions(combined_features)
                else:
                    final_positions = self.safe_holding_position

                self.move_to_position(final_positions)

                # ===== Shake measurement section =====
                now_perf = time.perf_counter()
                within_tol = False
                if (self.shake_measure_enabled and
                        not self.test_mode and
                        DynamixelSDK_available and
                        hasattr(self, 'packet_handler')):
                    cur = self._dxl_read_all_present_positions()
                    goal = self.last_positions
                    idx = min(self.shake_servo_index, len(goal) - 1)
                    if cur is not None and len(cur) > idx:
                        diff = abs(cur[idx] - int(goal[idx]))
                        within_tol = diff <= self.shake_tol_ticks

                if (self.shake_measure_enabled and within_tol and not self._prev_within_tol
                        and now_perf >= self._shake_cooldown_until):
                    self._start_shake_window(now_perf)

                self._prev_within_tol = within_tol

                if self._shake_active:
                    with self.accel_lock:
                        samples = [s for s in self.accel_queue if s[0] > self._shake_last_ts]
                    for ts, a_rms, g_rms in samples:
                        if a_rms > self._shake_max_a:
                            self._shake_max_a = a_rms
                        if a_rms < self._shake_min_a:
                            self._shake_min_a = a_rms
                        self._shake_sum_a += a_rms
                        self._shake_count += 1
                        if ts > self._shake_last_ts:
                            self._shake_last_ts = ts
                    if now_perf - self._shake_start_t >= self.shake_window_s:
                        self._finish_shake_window(now_perf)

                elapsed = time.time() - loop_start
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user")
        finally:
            self.cleanup()

    def start_hand_tracking_processes(self):
        for camera_name in ['left', 'right']:
            if self.cameras.get(camera_name) is not None:
                input_queue = getattr(self, f'mp_input_queue_{camera_name}')
                result_queue = getattr(self, f'mp_result_queue_{camera_name}')
                process = MPProcess(
                    target=hand_tracking_worker_process,
                    args=(camera_name, input_queue, result_queue,
                          self.mp_hand_filter_mode,
                          self.mp_processing_enabled,
                          self.mp_config),
                    daemon=True
                )
                process.start()
                self.mp_workers[camera_name] = process
                self.logger.info(f"Hand tracking process started for {camera_name} camera (PID: {process.pid})")

    # MARK: - Cleanup
    def cleanup(self):
        self.logger.info("Cleaning up resources...")
        self.running = False

        if not self.test_mode and DynamixelSDK_available:
            self.logger.info("Moving servos to home position...")
            try:
                self.move_to_position(DEFAULT_HOME_POSITION)
                time.sleep(1.0)
                self.disable_torque()
            except Exception as e:
                self.logger.error(f"Error during servo cleanup: {e}")

        if self.arduino_reader_thread and self.arduino_reader_thread.is_alive():
            self.arduino_reader_active = False
            self.arduino_reader_thread.join(timeout=2.0)

        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_active = False
            self.voice_thread.join(timeout=2.0)

        self.mp_processing_enabled.value = False
        for camera_name in ['left', 'right']:
            try:
                getattr(self, f'mp_input_queue_{camera_name}').put(None, timeout=1.0)
            except Exception:
                pass

        for camera_name, process in self.mp_workers.items():
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        self.queue_executor.shutdown(wait=True)
        self.stop_camera_threads()

        for camera in self.cameras.values():
            if camera is not None:
                try: camera.release()
                except Exception: pass

        if self.show_display:
            cv2.destroyAllWindows()

        self._mark_arduino_disconnected()

        if not self.test_mode and DynamixelSDK_available and hasattr(self, 'port_handler'):
            try:
                self.port_handler.closePort()
            except Exception:
                pass

        self.logger.info("Cleanup completed")

# MARK: - Entry point

def main():
    parser = argparse.ArgumentParser(description="Concurrent Voice + Hardware Runner (with Shake CSV, no RT)")
    parser.add_argument("--model", help="Path to trained model")
    parser.add_argument("--config", default="hardware_config.json")
    parser.add_argument("--arduino-port", default="/dev/arduino")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--display", action='store_true')
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--mic-index", type=int)
    parser.add_argument("--mic-hint", default=DEFAULT_MIC_HINT)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    if args.list_mics:
        list_input_devices()
        return

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

    runner.run_concurrent_loop()

if __name__ == "__main__":
    main()
