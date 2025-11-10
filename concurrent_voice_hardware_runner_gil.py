#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent Voice + Hardware Runner
- 카메라: 멀티프로세싱으로 MediaPipe 핸드 추적 → 좌표 스무딩 → 메인 프로세스로 전달
- 메인: 한 번 더 좌표 스무딩 → 모델 입력 → 로봇팔으로 부드럽게 전송
- 음성: 백그라운드 스레드
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
from typing import Optional, List, Tuple, Sequence, Callable
import multiprocessing as mp_module
from multiprocessing import Queue as MPQueue, Value as MPValue, Process as MPProcess

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
    #"TRACKING_ON": ["추적 시작", "시작", "추적 켜", "추적 온", "핸드 트래킹 켜", "손 추적 시작", "트래킹 시작", "트래킹 켜"],
    "TRACK_LEFT_HAND": ["왼손", "왼손 추적", "왼손만", "레프트 핸드", "왼쪽 손"],
    "TRACK_RIGHT_HAND": ["오른손", "오른손 추적", "오른손만", "라이트 핸드", "오른쪽 손"],
    "GO_HOME": ["고 홈", "홈", "홈으로", "집으로", "원위치", "제자리"],

    # Robot position control
    "STOP": ["스톱", "정지", "멈춰", "멈춰줘", "멈춰라"],
    #"EMERGENCY_RESET": ["리셋", "복구", "재시작", "긴급복구", "리셋해줘"],

    #"EXIT": ["종료", "종로"]
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
DEFAULT_HOME_POSITION = [2048, 3328, 1140, 1600, 2048]

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
    """Feedforward network that mirrors training-time architecture."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 5,
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
    if quick_contains(text, ["추적 시작", "트래킹 시작"]):
        return "TRACKING_ON"

    priority_commands = [
        #"EXIT", 
        #"EMERGENCY_RESET", 
        "STOP", "GO_HOME",
        #"TRACKING_ON", 
        "TRACK_LEFT_HAND", "TRACK_RIGHT_HAND",
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
                except (OSError, IOError) as e:
                    # Hardware or I/O error with this configuration
                    if self._debug:
                        print(f"  DEBUG: Failed to open mic with rate={r}, ch={ch}: {e}")
                    last_err = e
                    continue
                except Exception as e:
                    # Unexpected error
                    if self._debug:
                        print(f"  DEBUG: Unexpected error with rate={r}, ch={ch}: {e}")
                    last_err = e
                    continue
        raise RuntimeError(f"마이크 열기 실패: {last_err}")

    def __exit__(self, *args):
        self.closed = True
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception as e:
                if self._debug:
                    print(f"  DEBUG: Failed to stop stream: {e}")
            try:
                self._stream.close()
            except Exception as e:
                if self._debug:
                    print(f"  DEBUG: Failed to close stream: {e}")
        try:
            self._buff.put_nowait(None)
        except queue.Full:
            pass  # Queue full is acceptable here
        except Exception as e:
            if self._debug:
                print(f"  DEBUG: Failed to put None to buffer: {e}")
        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                if self._debug:
                    print(f"  DEBUG: Failed to terminate PyAudio: {e}")

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

# MARK: - Multiprocessing Worker (with smoothing)

def hand_tracking_worker_process(camera_name, input_queue, result_queue, hand_filter_mode, processing_enabled, config):
    """
    Isolated worker process for hand tracking
    - MediaPipe로 손 추적
    - 캐시 + 스무딩 + 데드존 + 점프 처리
    """
    # Configure logging for this worker process
    import logging
    logger = logging.getLogger(f"worker.{camera_name}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s'))
    logger.addHandler(handler)

    logger.info(f"Hand tracking worker started for {camera_name} camera")

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

    # 스무딩용 상태값
    smoothed_xy = None
    smooth_alpha = 1.0      # EMA 계수
    deadzone_px = 4.0        # 이 픽셀 이하면 떨림으로 보고 무시
    jump_px = 60.0           # 이 이상이면 손을 확 옮긴 걸로 보고 바로 점프

    def cache_and_fill(xy):
        nonlocal coord_cache
        now = time.time()
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
            frame = input_queue.get(timeout=0.1)

            if frame is None:
                break

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
                    # Send NaN to result queue to signal error
                    try:
                        result_queue.put_nowait(([np.nan, np.nan], None))
                    except queue.Full:
                        pass  # Queue full is acceptable, will retry next frame

            # 1차: 캐시로 NaN 보정
            xy_final = cache_and_fill(raw_xy)

            # 2차: 스무딩
            if smoothed_xy is None or not np.isfinite(smoothed_xy).all():
                smoothed_xy = xy_final
            else:
                new_x, new_y = xy_final
                old_x, old_y = smoothed_xy
                if np.isfinite([new_x, new_y]).all():
                    dist = ((new_x - old_x)**2 + (new_y - old_y)**2) ** 0.5
                    if dist < deadzone_px:
                        # 너무 작은 떨림은 무시
                        pass
                    elif dist > jump_px:
                        # 확 옮겼으면 즉시 반영
                        smoothed_xy = [new_x, new_y]
                    else:
                        # 일반적인 경우 EMA
                        smoothed_xy = [
                            smooth_alpha * new_x + (1 - smooth_alpha) * old_x,
                            smooth_alpha * new_y + (1 - smooth_alpha) * old_y,
                        ]
                # 새 값이 NaN이면 이전 스무딩값 유지

            # 결과 전송
            try:
                result_queue.put_nowait((smoothed_xy, last_handedness))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                    result_queue.put_nowait((smoothed_xy, last_handedness))
                except (queue.Full, queue.Empty):
                    pass  # Queue operation failed, will retry next frame

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)

    hands_processor.close()
    logger.info(f"Hand tracking worker shutdown for {camera_name}")

# MARK: - Main System Class

class ConcurrentVoiceHardwareRunner:
    def __init__(self, model_path=None, hardware_config_path='hardware_config.json',
                 arduino_port="/dev/arduino", test_mode=False, target_fps=60.0,
                 show_display=False, mic_index=None, mic_hint="blue", debug=False):

        self._last_frames = {}

        # System control flags
        self.hand_tracking_enabled = False
        self.voice_active = True
        self.running = True
        self.robot_stopped = False

        # MediaPipe 프레임 스킵 기본값
        self.mediapipe_process_every_n_frames = 1

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
        self._frame_queue_lock = threading.RLock()
        self._robot_state_lock = threading.RLock()
        self._mediapipe_lock = threading.RLock()

        # Logging
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
        self.capture_active = threading.Event()
        self.cameras = {}

        self.setup_cameras()
        self.setup_mediapipe()
        if self.show_display:
            self.setup_display_windows()

        # Servos
        if not test_mode and DynamixelSDK_available:
            self.setup_servos()
        else:
            self.logger.info("🧪 TEST MODE - Hardware control disabled")
            self.setup_servo_defaults()

        # Safety / smoothing
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_successful_positions = DEFAULT_HOME_POSITION.copy()

        # 이 버전에서는 모터 측 스텝 제한으로 부드럽게 할 수도 있지만,
        # 여기서는 원래 코드 흐름을 그대로 두고 입력 쪽을 더 부드럽게 한다.
        self.position_smoothing_alpha = 0.1

        self.last_positions = DEFAULT_HOME_POSITION.copy()
        self.emergency_stop = False
        self.safe_zone_min = [1024, 1900, 1024, 1024, 512]
        self.safe_zone_max = [2944, 3520, 3340, 3136, 4096]

        self.safe_holding_position = DEFAULT_HOME_POSITION.copy()
        self.last_sent_positions = None

        # 입력 좌표에 대한 메인단 스무딩 상태
        self.prev_left_xy = None
        self.prev_right_xy = None
        self.input_smooth_alpha = 1.0   # 메인단 EMA
        self.input_deadzone_px = 2.0    # 메인단 데드존

        # Cache TTL for worker processes
        self.cache_ttl = 0.0

        # 메인 프로세스의 좌표 캐시 (Queue Empty 대응)
        self.main_coord_cache = {
            'left': {'xy': [np.nan, np.nan], 't': 0.0},
            'right': {'xy': [np.nan, np.nan], 't': 0.0},
        }

        # Multiprocessing queues
        self.mp_input_queue_left = MPQueue(maxsize=2)
        self.mp_input_queue_right = MPQueue(maxsize=2)
        self.mp_result_queue_left = MPQueue(maxsize=2)
        self.mp_result_queue_right = MPQueue(maxsize=2)

        # ThreadPoolExecutor for parallel queue processing
        self.queue_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="queue_processor")

        # Shared control flags
        self.mp_hand_filter_mode = MPValue('i', 0)  # 0=None, 1=Left, 2=Right
        self.mp_processing_enabled = MPValue('b', True)

        # Worker processes container
        self.mp_workers = {}

        # MediaPipe config
        self.mp_config = {
            'min_detection_confidence': 0.8,
            'min_tracking_confidence': 0.8,
            'model_complexity': 1,
            'process_every_n_frames': self.mediapipe_process_every_n_frames,
            'cache_ttl': self.cache_ttl,
            'show_display': self.show_display
        }

        # Initialize Arduino
        self.ensure_arduino_connected(force=True)

        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    # MARK: - Signal Handler

    def _signal_handler(self, signum, frame):
        """시그널 수신 시 정상 종료 처리"""
        sig_name = signal.Signals(signum).name
        self.logger.info(f"Received signal {sig_name}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)

    # MARK: - Arduino Management

    def ensure_arduino_connected(self, force: bool = False) -> bool:
        """
        Ensure Arduino is connected, with thread-safe connection establishment.

        Args:
            force: If True, retry connection even if retry interval hasn't elapsed

        Returns:
            True if Arduino is connected and ready, False otherwise
        """
        if self.test_mode:
            return False

        with self._arduino_lock:
            # Check existing connection
            if self.arduino and getattr(self.arduino, "is_open", False):
                return True

            # Check retry interval
            now = time.time()
            if not force and (now - self._last_arduino_attempt) < self._arduino_retry_interval:
                return False

            self._last_arduino_attempt = now

            # Close any existing connection
            if self.arduino:
                try:
                    self.arduino.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close existing Arduino connection: {e}")
                self.arduino = None

            # Attempt new connection (still under lock to prevent race condition)
            try:
                ser = serial.Serial(self.arduino_port, 9600, timeout=1, write_timeout=1)
                time.sleep(2)  # Arduino reset delay
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception as e:
                    self.logger.warning(f"Failed to reset Arduino buffers: {e}")

                self.arduino = ser
                self._last_arduino_ping = time.time()
                self.logger.info(f"✓ Arduino connected ({self.arduino_port})")
                return True

            except Exception as e:
                self.logger.error(f"❌ Arduino connection failed: {e}")
                self.arduino = None
                self._last_arduino_ping = 0.0
                return False

    def _mark_arduino_disconnected(self) -> None:
        with self._arduino_lock:
            if self.arduino:
                try:
                    self.arduino.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close Arduino during disconnect: {e}")
            self.arduino = None
            self._last_arduino_attempt = 0.0
            self._last_arduino_ping = 0.0

    def send_arduino_command(self, cmd: str, quiet: bool = False) -> bool:
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

    # MARK: - Model Loading

    def load_model(self):
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
                        output_dim=5,
                        hidden_sizes=hidden_sizes,
                        dropout=dropout
                    ).to(self.device)
                    self.logger.info(
                        "Created ConfigurableFeedforward model with hidden_sizes=%s, dropout=%.3f",
                        hidden_sizes,
                        dropout
                    )

            elif 'res' in arch_lower:
                self.model = ResFeedforward(
                    input_dim=4,
                    output_dim=5,
                    dropout=config.get('dropout', 0.0)
                ).to(self.device)
                self.logger.info(f"Created ResFeedforward model: {arch}")

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
                self.logger.info("Created SimpleTransformer model")

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
            self.config = {}

    # MARK: - Camera Setup

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

    # MARK: - MediaPipe Setup

    def setup_mediapipe(self):
        self.mp_drawing = mp.solutions.drawing_utils

    # MARK: - Display Setup

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

    # MARK: - Servo Setup

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
        if self.test_mode or not DynamixelSDK_available:
            return True

        if self.emergency_stop:
            self.logger.warning("Emergency stop active - not sending commands")
            return False

        if self.last_sent_positions is not None:
            positions_changed = False
            for i, pos in enumerate(positions):
                if i >= len(self.last_sent_positions) or abs(pos - self.last_sent_positions[i]) > 1:
                    positions_changed = True
                    break
            if not positions_changed:
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

    # MARK: - Camera Threading

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
                    with self._frame_queue_lock:
                        try:
                            self.frame_queues[camera_name].put_nowait(frame)
                        except queue.Full:
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
        self.capture_active.clear()
        for camera_name, thread in self.camera_threads.items():
            thread.join(timeout=1.0)
            if thread.is_alive():
                self.logger.warning(f"Camera thread {camera_name} still alive after join timeout")
        self.camera_threads.clear()

    def start_hand_tracking_processes(self):
        for camera_name in ['left', 'right']:
            if self.cameras.get(camera_name) is not None:
                input_queue = getattr(self, f'mp_input_queue_{camera_name}')
                result_queue = getattr(self, f'mp_result_queue_{camera_name}')

                process = MPProcess(
                    target=hand_tracking_worker_process,
                    args=(
                        camera_name,
                        input_queue,
                        result_queue,
                        self.mp_hand_filter_mode,
                        self.mp_processing_enabled,
                        self.mp_config
                    ),
                    daemon=True
                )
                process.start()
                self.mp_workers[camera_name] = process
                self.logger.info(f"Hand tracking process started for {camera_name} camera (PID: {process.pid})")

    def get_latest_frames(self):
        frames = {}
        for camera_name in self.cameras.keys():
            try:
                with self._frame_queue_lock:
                    try:
                        frame = self.frame_queues[camera_name].get_nowait()
                        self._last_frames[camera_name] = frame
                    except Empty:
                        frame = self._last_frames.get(camera_name, None)
                frames[camera_name] = frame
            except KeyError:
                frames[camera_name] = None
        return frames


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

    # MARK: - Input XY smoothing (main process)

    def _smooth_xy(self, xy, is_left=True):
        """메인 프로세스에서 한 번 더 좌표를 부드럽게"""
        arr = np.array(xy, dtype=np.float32)

        if is_left:
            prev = self.prev_left_xy
        else:
            prev = self.prev_right_xy

        # NaN이 들어오면 이전 유효 값 반환 (깜빡임 방지)
        if not np.isfinite(arr).all():
            if prev is not None and np.isfinite(prev).all():
                return prev.tolist() if isinstance(prev, np.ndarray) else prev
            return xy  # 이전 값도 없으면 NaN 그대로

        if prev is None or not np.isfinite(prev).all():
            new_xy = arr
        else:
            # 데드존
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

    # MARK: - Display Update

    def update_display(self, frames, left_features, right_features, left_handedness=None, right_handedness=None):
        if not self.show_display:
            return

        for camera_name, frame in frames.items():
            if frame is not None:
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

                filter_val = self.mp_hand_filter_mode.value
                if filter_val != 0:
                    filter_mode_str = "Left" if filter_val == 1 else "Right"
                    filter_text = f"{filter_mode_str} Only"
                    cv2.rectangle(display_frame, (210, 10), (380, 50), (0, 0, 0), -1)
                    cv2.putText(display_frame, filter_text,
                               (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                if self.emergency_stop:
                    cv2.rectangle(display_frame, (5, 5), (635, 475), (0, 0, 255), 5)
                    cv2.putText(display_frame, "EMERGENCY!",
                               (200, 240), cv2.FONT_HERSHEY_BOLD, 1.5, (0, 0, 255), 4)

                if self.hand_tracking_enabled:
                    features = left_features if camera_name == 'left' else right_features
                    handedness = left_handedness if camera_name == 'left' else right_handedness
                    x, y = features
                    if np.isfinite([x, y]).all():
                        color = (0, 255, 0) if camera_name == 'left' else (0, 255, 255)
                        cv2.circle(display_frame, (int(x), int(y)), 10, color, -1)

                        # if handedness:
                        #     hand_text = f"{handedness} Hand"
                        #     cv2.putText(display_frame, hand_text,
                        #                (int(x) + 15, int(y) - 10),
                        #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                window_name = f"{camera_name.title()} Camera"
                cv2.imshow(window_name, display_frame)

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
                    if text and detect_wake_interim(text):
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
        print("📖 Arduino reader thread started")
        while self.arduino_reader_active and self.running:
            try:
                with self._arduino_lock:
                    if self.arduino and getattr(self.arduino, "is_open", False):
                        if self.arduino.in_waiting > 0:
                            try:
                                line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                                if line and self.debug:
                                    print(f"[Arduino] {line}")
                            except Exception as e:
                                if self.debug:
                                    self.logger.error(f"Arduino read error: {e}")
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Arduino reader thread error: {e}")
                time.sleep(1.0)
        print("📖 Arduino reader thread stopped")

    # MARK: - Command Handling

    def handle_voice_command(self, command):
        print(f"🗣️ Voice Command: {command}")

        if command == "EXIT":
            print("🛑 Exit command received")
            self.running = False

        elif command == "EMERGENCY_RESET":
            print("🔄 EMERGENCY RESET - Recovering system...")
            with self._robot_state_lock:
                if self.emergency_stop:
                    if not self.test_mode:
                        safe_home = self.safe_holding_position
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

        elif command == "STOP":
            with self._robot_state_lock:
                self.robot_stopped = True
                self.hand_tracking_enabled = False
            self.send_arduino_command("LED_EFFECT:11", quiet=True)
            print("⏹️ Robot STOPPED at current position - Hand tracking disabled")

        elif command == "GO_HOME":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = False
                self.consecutive_failures = 0
            self.send_arduino_command("LED_EFFECT:10", quiet=True)
            print("🏠 Going HOME - Moving to safe position")
            if not self.test_mode:
                self.move_to_position(self.safe_holding_position)

        elif command == "TRACKING_ON":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = True
            self.mp_hand_filter_mode.value = 0
            self.send_arduino_command("LED_EFFECT:9", quiet=True)
            print("📷 Hand tracking ENABLED - Robot will follow hand movements")

        elif command == "TRACK_LEFT_HAND":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = True
            self.mp_hand_filter_mode.value = 1
            print("👈 Tracking LEFT hand only - Right hand will be ignored")

        elif command == "TRACK_RIGHT_HAND":
            with self._robot_state_lock:
                self.robot_stopped = False
                self.hand_tracking_enabled = True
            self.mp_hand_filter_mode.value = 2
            print("👉 Tracking RIGHT hand only - Left hand will be ignored")

        elif command in LED_COMMAND_MAP:
            arduino_cmd = LED_COMMAND_MAP[command]
            if self.ensure_arduino_connected(force=True):
                success = self.send_arduino_command(arduino_cmd)
                if not success:
                    print(f"⚠️ Arduino 전송 실패 - 연결을 재시도합니다 ({arduino_cmd})")
                    self._mark_arduino_disconnected()
            else:
                print(f"⚠️ Arduino not connected, cannot send {arduino_cmd}")

    # MARK: - Queue Processing Helper

    def _process_camera_queue(self, camera_name, frame, input_queue, result_queue):
        """단일 카메라의 큐 처리 (프레임 전송 + 결과 수신)"""
        # 프레임을 input queue에 넣기
        if frame is not None:
            try:
                input_queue.put_nowait(frame)
            except queue.Full:
                pass

        # 결과 큐에서 가져오기
        features = [np.nan, np.nan]
        handedness = None

        try:
            result = result_queue.get(timeout=0.001)
            features, handedness = result
            # 유효한 값이면 캐시 업데이트
            if np.isfinite(features).all():
                self.main_coord_cache[camera_name]['xy'] = features
                self.main_coord_cache[camera_name]['t'] = time.time()
        except queue.Empty:
            # Queue가 비었으면 캐시된 값 사용
            cached = self.main_coord_cache[camera_name]
            if self.cache_ttl <= 0.0 or (time.time() - cached['t']) <= self.cache_ttl:
                features = cached['xy']

        return features, handedness

    # MARK: - Main Loop

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
                    # 병렬로 두 카메라 큐 처리
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

                    # 두 결과를 동시에 수집
                    left_features, left_handedness = left_future.result()
                    right_features, right_handedness = right_future.result()

                    # 메인단에서도 한 번 더 스무딩 (이제 항상 호출, NaN 처리는 _smooth_xy 내부에서)
                    left_features = self._smooth_xy(left_features, is_left=True)
                    right_features = self._smooth_xy(right_features, is_left=False)

                else:
                    left_features = [np.nan, np.nan]
                    right_features = [np.nan, np.nan]
                    left_handedness = None
                    right_handedness = None

                if self.show_display:
                    self.update_display(frames, left_features, right_features, left_handedness, right_handedness)

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
        """타임아웃 보호가 있는 cleanup 메서드 - 최대 10초 내에 종료 보장"""
        import os

        self.logger.info("Starting cleanup with 10-second timeout protection...")

        # _do_cleanup을 별도 스레드에서 실행
        cleanup_thread = threading.Thread(target=self._do_cleanup, name="CleanupThread")
        cleanup_thread.daemon = True
        cleanup_thread.start()

        # 10초 타임아웃으로 대기
        cleanup_thread.join(timeout=10.0)

        if cleanup_thread.is_alive():
            self.logger.error("⚠️  Cleanup timed out after 10 seconds, forcing exit...")
            self.logger.error("Some resources may not be properly released")
            os._exit(1)  # 강제 종료
        else:
            self.logger.info("✓ Cleanup completed successfully within timeout")

    def _do_cleanup(self):
        """실제 cleanup 작업을 수행하는 내부 메서드"""
        self.logger.info("Cleaning up resources...")
        self.running = False

        if not self.test_mode and DynamixelSDK_available:
            self.logger.info("Moving servos to home position...")
            try:
                if self.move_to_position(DEFAULT_HOME_POSITION):
                    time.sleep(1.0)
                    self.logger.info("Servos moved to home position")
                else:
                    self.logger.warning("Failed to move servos to home position")

                self.disable_torque()
                self.logger.info("Servo torque disabled")

            except Exception as e:
                self.logger.error(f"Error during servo cleanup: {e}")

        if self.arduino_reader_thread and self.arduino_reader_thread.is_alive():
            self.arduino_reader_active = False
            self.arduino_reader_thread.join(timeout=2.0)
            if self.arduino_reader_thread.is_alive():
                self.logger.warning("Arduino reader thread still alive after join timeout")

        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_active = False  # 명시적으로 음성 스레드 종료 신호
            self.voice_thread.join(timeout=2.0)
            if self.voice_thread.is_alive():
                self.logger.warning("Voice thread still alive after join timeout")

        if hasattr(self, 'mp_workers') and self.mp_workers:
            self.logger.info("Shutting down hand tracking processes...")
            self.mp_processing_enabled.value = False

            for camera_name in ['left', 'right']:
                try:
                    queue_obj = getattr(self, f'mp_input_queue_{camera_name}', None)
                    if queue_obj:
                        queue_obj.put(None, timeout=1.0)
                except Exception:
                    pass

            for camera_name, process in self.mp_workers.items():
                process.join(timeout=2.0)
                if process.is_alive():
                    self.logger.warning(f"Force terminating {camera_name} hand tracking process")
                    process.terminate()
                    process.join(timeout=1.0)

            self.logger.info("Hand tracking processes shutdown")

        # Shutdown ThreadPoolExecutor
        if hasattr(self, 'queue_executor'):
            self.logger.info("Shutting down queue executor...")
            try:
                self.queue_executor.shutdown(wait=True, timeout=2.0)
                self.logger.info("Queue executor shutdown complete")
            except Exception as e:
                self.logger.warning(f"Error during queue executor shutdown: {e}")

        # Clean up MPQueues to prevent hanging
        self.logger.info("Cleaning up MPQueues...")
        for queue_name in ['mp_input_queue_left', 'mp_input_queue_right',
                          'mp_result_queue_left', 'mp_result_queue_right']:
            try:
                queue_obj = getattr(self, queue_name, None)
                if queue_obj:
                    queue_obj.close()
                    queue_obj.join_thread()
                    self.logger.debug(f"Closed {queue_name}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {queue_name}: {e}")

        self.stop_camera_threads()

        for camera in self.cameras.values():
            if camera is not None:
                try:
                    camera.release()
                except Exception as e:
                    self.logger.error(f"Failed to release camera: {e}")

        if self.show_display:
            cv2.destroyAllWindows()

        with self._arduino_lock:
            arduino_ref = self.arduino if self.arduino and getattr(self.arduino, "is_open", False) else None

        if arduino_ref:
            try:
                self.logger.info("Turning off Arduino LED...")
                if self.send_arduino_command("OFF"):
                    time.sleep(0.5)
                    self.logger.info("Arduino LED turned off")
            except Exception as e:
                self.logger.error(f"Failed to turn off Arduino LED: {e}")

        self._mark_arduino_disconnected()

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
    print("  Robot control: '정지' / '홈'")
    print("  Arduino LEDs: '켜' / '꺼'")
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
