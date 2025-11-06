#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent Voice + Hardware Runner (legacy base)
+ Reaction-time measurement (observation only)
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
from typing import Optional, List, Tuple, Sequence

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
    "LED_OFF": ["꺼", "꺼줘", "불꺼", "불 꺼", "라이트오프", "라이트 오프", "끄자"],
    "LED_ON": ["켜", "켜줘", "불켜", "불 켜", "라이트온", "라이트 온", "키자"],
    "TRACKING_ON": ["추적 시작", "시작", "추적 켜", "추적 온", "핸드 트래킹 켜", "손 추적 시작", "트래킹 시작", "트래킹 켜"],
    "GO_HOME": ["고 홈", "홈", "홈으로", "집으로", "원위치", "제자리", "home"],
    "STOP": ["스톱", "정지", "멈춰", "stop"],
    "EXIT": ["그만"]
}

# =============================================================================
# PYTORCH MODEL CLASSES
# =============================================================================

class SimpleTransformer(nn.Module):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.network(x)

class ConfigurableFeedforward(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, hidden_sizes: Sequence[int] | None=None, dropout: float=0.0):
        super().__init__()
        hs = tuple(hidden_sizes) if hidden_sizes else ()
        if not hs: raise ValueError("hidden_sizes must contain at least one layer")
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hs:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            if dropout > 0.0: layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.network(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        r = self.proj(x)
        y = self.fc1(x); y = self.act(y); y = self.fc2(y); y = self.drop(y)
        y = self.norm(y + r)
        return y

class ResFeedforward(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, dropout=0.0):
        super().__init__()
        self.a = ResidualBlock(input_dim, 8, 16, dropout)
        self.b = ResidualBlock(16, 8, output_dim, dropout)
        self.skip = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.b(self.a(x)) + self.skip(x)

# =============================================================================
# VOICE / MIC UTILS (same as before, omitted for brevity changes)
# =============================================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).lower()

def quick_contains(text: str, keys: List[str]) -> bool:
    s = normalize(text); return any(normalize(k) in s for k in keys)

def detect_wake_interim(text: str) -> bool:
    return quick_contains(text, [WAKE_CANONICAL] + WAKE_VARIANTS)

def detect_cmd_interim(text: str) -> Optional[str]:
    if quick_contains(text, COMMAND_SYNONYMS["EXIT"]): return "EXIT"
    if quick_contains(text, COMMAND_SYNONYMS["STOP"]): return "STOP"
    if quick_contains(text, COMMAND_SYNONYMS["GO_HOME"]): return "GO_HOME"
    if quick_contains(text, ["추적 시작", "트래킹 시작"]): return "TRACKING_ON"
    if quick_contains(text, COMMAND_SYNONYMS["TRACKING_ON"]): return "TRACKING_ON"
    if quick_contains(text, COMMAND_SYNONYMS["LED_OFF"]): return "LED_OFF"
    if quick_contains(text, COMMAND_SYNONYMS["LED_ON"]): return "LED_ON"
    return None

def list_input_devices():
    p = pyaudio.PyAudio()
    print("=== Input devices ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            print(f"[{i}] {info.get('name')} (in={info.get('maxInputChannels')}, rate={int(info.get('defaultSampleRate',0))})")
    p.terminate()

def pick_device_index(p: pyaudio.PyAudio, index: Optional[int], hint: str) -> int:
    if index is not None: return index
    chosen = None; hint_l = (hint or "").lower()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) <= 0: continue
        name = (info.get("name", "") or "").lower()
        if hint_l and hint_l in name: return i
        if chosen is None: chosen = i
    return chosen if chosen is not None else 0

class MicrophoneStream:
    def __init__(self, mic_index: Optional[int], mic_hint: str, debug: bool):
        self.mic_index = mic_index; self.mic_hint = mic_hint; self.debug = debug
        self._pa=None; self._stream=None; self._buff=queue.Queue(maxsize=100)
        self._carry=b""; self._ratecv_state=None; self._hw_rate=None; self._hw_channels=1
        self.closed=True; self.vad=webrtcvad.Vad(1)
    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        device_index = pick_device_index(self._pa, self.mic_index, self.mic_hint)
        dinfo = self._pa.get_device_info_by_index(device_index)
        default_rate = int(dinfo.get("defaultSampleRate", 48000))
        rate_candidates = [16000, default_rate, 48000, 44100, 32000]
        last_err=None
        for ch in (1,2):
            for r in rate_candidates:
                try:
                    fpb = int(r*FRAME_MS/1000)
                    self._stream = self._pa.open(format=pyaudio.paInt16, channels=ch, rate=r, input=True,
                                                 input_device_index=device_index, frames_per_buffer=fpb,
                                                 stream_callback=self._fill_buffer)
                    self._hw_rate, self._hw_channels = r, ch
                    self.closed=False
                    print(f"🎤 Mic: [{device_index}] {dinfo.get('name')} @ {r} Hz, ch={ch}")
                    return self
                except Exception as e:
                    last_err=e; continue
        raise RuntimeError(f"마이크 열기 실패: {last_err}")
    def __exit__(self,*args):
        self.closed=True
        if self._stream:
            try:self._stream.stop_stream()
            except: pass
            try:self._stream.close()
            except: pass
        try:self._buff.put_nowait(None)
        except: pass
        if self._pa: self._pa.terminate()
    def _fill_buffer(self, in_data,*_):
        try:
            if self._buff.full(): self._buff.get_nowait()
            self._buff.put_nowait(in_data)
        except queue.Full: pass
        return (None, pyaudio.paContinue)
    def _to_mono_16k(self, data: bytes) -> bytes:
        pcm=data
        if self._hw_channels==2:
            try: pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
            except Exception:
                mono=bytearray()
                for (l,r) in struct.iter_unpack('<hh', pcm):
                    mono.extend(struct.pack('<h', int((l+r)/2)))
                pcm=bytes(mono)
        if self._hw_rate!=TARGET_RATE:
            pcm, self._ratecv_state = audioop.ratecv(pcm, 2, 1, self._hw_rate, TARGET_RATE, self._ratecv_state)
        return pcm
    def generator(self):
        while not self.closed:
            try: chunk=self._buff.get(timeout=1.0)
            except queue.Empty: continue
            if chunk is None: return
            yield self._to_mono_16k(chunk)

def build_client_and_config(single_utter: bool):
    client = speech.SpeechClient()
    phrases = list(set([WAKE_CANONICAL] + WAKE_VARIANTS + sum(COMMAND_SYNONYMS.values(), [])))
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
        config=config, interim_results=True, single_utterance=single_utter
    )
    return client, streaming_config

def start_stream(mic_index: Optional[int], mic_hint: str, debug: bool, for_command: bool):
    client, streaming_config = build_client_and_config(single_utter=for_command)
    stream = MicrophoneStream(mic_index, mic_hint, debug); stream.__enter__()
    audio_gen = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=f) for f in audio_gen)
    responses = client.streaming_recognize(streaming_config, requests)
    return stream, responses

# =============================================================================
# ARDUINO (unchanged helpers)
# =============================================================================

def open_arduino(port: str, baud: int = 9600):
    try:
        ser = serial.Serial(port, baud, timeout=1); time.sleep(2)
        print(f"✓ Arduino 연결 성공! ({port})"); return ser
    except Exception as e:
        print(f"❌ Arduino 연결 실패: {e}"); return None

def send_arduino(ser, cmd: str):
    try:
        ser.write((cmd+"\n").encode()); ser.flush()
        print(f"👉 Arduino: {cmd}")
    except Exception as e:
        print(f"❌ Arduino 전송 실패: {e}")

# =============================================================================
# MAIN CONCURRENT HARDWARE RUNNER
# =============================================================================

class ConcurrentVoiceHardwareRunner:
    # ----------- Reaction-time settings -----------
    RT_BATCH = 20               # 유효 사이클 20개 단위로 요약 출력
    RT_MOVE_TIMEOUT = 0.300     # 움직임 감지 최대 대기(초)
    RT_POLL_INTERVAL = 0.005    # present velocity 폴링 간격(초)
    DXL_ADDR_PRESENT_VELOCITY = 128  # X-series address (4 bytes)

    def __init__(self, model_path=None, hardware_config_path='hardware_config.json',
                 arduino_port="/dev/arduino", test_mode=False, target_fps=60.0,
                 show_display=False, mic_index=None, mic_hint="blue", debug=False):

        self.hand_tracking_enabled = False
        self.voice_active = True
        self.running = True
        self.robot_stopped = False

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

        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self.model = None; self.model_type=None
        self.scaler_X=None; self.scaler_y=None; self.normalize=False
        if model_path: self.load_model()

        self.load_hardware_config(hardware_config_path)

        self.frame_queues = {}; self.camera_threads = {}; self.capture_active=False
        self.cameras = {}; self.setup_cameras()
        self.setup_mediapipe()
        if self.show_display: self.setup_display_windows()

        if not test_mode and DynamixelSDK_available:
            self.setup_servos()
        else:
            self.logger.info("🧪 TEST MODE - Hardware control disabled")
            self.setup_servo_defaults()

        self.frame_count=0; self.total_inference_time=0.0; self.last_fps_time=time.time()
        self.consecutive_failures=0; self.max_consecutive_failures=10
        self.last_successful_positions=[2048,3328,1140,3072]
        self.position_smoothing_alpha=0.3
        self.last_positions=[2048,3328,1140,3072]
        self.emergency_stop=False
        self.safe_zone_min=[1280,1920,1120,1664]
        self.safe_zone_max=[2944,3456,3200,3136]
        self.safe_holding_position=[2048,3328,1140,3072]

        # ---- Reaction-time buffers ----
        self.rt_cam_to_pred: List[float] = []
        self.rt_cam_to_tx:   List[float] = []
        self.rt_cam_to_move: List[float] = []  # may have NaN if not measurable
        self.rt_pred_to_tx:  List[float] = []

        self.arduino = open_arduino(self.arduino_port)

    # --------- Model / HW config loaders (unchanged) ----------
    def load_model(self):
        self.model_type=None; self.normalize=False; self.scaler_X=None; self.scaler_y=None
        if self.model_path.lower().endswith(".joblib"):
            self.logger.info(f"Loading XGBoost model from {self.model_path}")
            self.model = joblib.load(self.model_path); self.model_type="xgb"
            self.logger.info("XGBoost model loaded (NaN inputs supported)."); return
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = ckpt.get('model_config', {}) or {}; sd = ckpt['model_state_dict']
            arch = (config.get('arch') or
                    ('feedforward' if 'hidden_sizes' in config else
                     ('resfeedforward' if 'resfeedforward' in config.get('model_name','').lower() else 'transformer')))
            al = arch.lower()
            if 'feedforward' in al and 'res' not in al:
                hs = config.get('hidden_sizes'); has_ln = any('.1.' in k for k in sd.keys())
                if not hs and not has_ln:
                    self.model = SimpleTransformer(input_dim=4, output_dim=4,
                                                   d_model=config.get('d_model', config.get('dim_feedforward',12)),
                                                   nhead=config.get('nhead',1),
                                                   num_layers=1,
                                                   dim_feedforward=config.get('dim_feedforward',12),
                                                   dropout=0.0).to(self.device)
                    self.logger.info("Detected legacy feedforward checkpoint; using SimpleTransformer layout")
                else:
                    if not hs:
                        n=max(int(config.get('num_layers',1)),1); d=int(config.get('dim_feedforward',12))
                        hs=tuple(d for _ in range(n))
                    else:
                        hs=tuple(int(x) for x in hs)
                    self.model = ConfigurableFeedforward(input_dim=4, output_dim=4,
                                                         hidden_sizes=hs,
                                                         dropout=float(config.get('dropout',0.0))).to(self.device)
                    self.logger.info(f"Created ConfigurableFeedforward model with hidden_sizes={hs}")
            elif 'res' in al:
                self.model = ResFeedforward(input_dim=4, output_dim=4,
                                            dropout=config.get('dropout',0.0)).to(self.device)
                self.logger.info("Created ResFeedforward model")
            else:
                self.model = SimpleTransformer(input_dim=4, output_dim=4,
                                               d_model=config.get('d_model',8),
                                               nhead=config.get('nhead',1),
                                               num_layers=config.get('num_layers',1),
                                               dim_feedforward=config.get('dim_feedforward',12),
                                               dropout=0.0).to(self.device)
                self.logger.info("Created SimpleTransformer model")
            self.model.load_state_dict(ckpt['model_state_dict']); self.model.eval()
            self.scaler_X = ckpt.get('scaler_X'); self.scaler_y = ckpt.get('scaler_y')
            self.normalize = ckpt.get('normalize', False)
            self.model_type="torch"
            self.logger.info(f"PyTorch model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}"); raise

    def load_hardware_config(self, config_path):
        try:
            with open(config_path,'r') as f: self.config=json.load(f)
            self.logger.info(f"Hardware config loaded: {config_path}")
        except FileNotFoundError:
            self.logger.warning(f"Hardware config not found: {config_path}, using defaults")
            self.config={"servos":{"port":"/dev/ttyUSB0","baudrate":1000000,"ids":[1,2,3,4],
                                   "min_positions":[1280,1920,1120,1664],"max_positions":[2944,3456,3200,3136]},
                         "cameras":{"left":{"id":0,"width":640,"height":480},
                                    "right":{"id":2,"width":640,"height":480}}}

    # --------- Cameras / MediaPipe (unchanged except logs) ----------
    def setup_cameras(self):
        self.cameras={}
        cfg=self.config.get('cameras',{})
        lc=cfg.get('cam_left',{'id':0,'enabled':True})
        self.cameras['left'] = self._setup_single_camera('left', lc.get('id',0), lc) if lc.get('enabled',True) else None
        rc=cfg.get('cam_right',{'id':2,'enabled':True})
        self.cameras['right']= self._setup_single_camera('right', rc.get('id',2), rc) if rc.get('enabled',True) else None
    def _setup_single_camera(self, name, cam_id, config):
        try:
            cam=cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            if not cam.isOpened(): cam=cv2.VideoCapture(cam_id)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH,640); cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                cam.set(cv2.CAP_PROP_FPS,30); cam.set(cv2.CAP_PROP_BUFFERSIZE,1)
                self.logger.info(f"{name.title()} camera ready (id={cam_id})"); return cam
            else:
                self.logger.warning(f"{name.title()} camera not available (id={cam_id})"); return None
        except Exception as e:
            self.logger.error(f"Failed to setup {name} camera: {e}"); return None

    def setup_mediapipe(self):
        """Setup MediaPipe hand tracking (version-safe; use keyword args)"""
        self.mp_hands = mp.solutions.hands

        # 공통 권장값: 모델 단순화(0), 탐지/추적 신뢰도 낮춰 빠르게
        common_kwargs = dict(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,            # <-- int 로 명시 (0 or 1)
            min_detection_confidence=0.3,  # <-- float
            min_tracking_confidence=0.3    # <-- float
        )

        try:
            self.hands_left = self.mp_hands.Hands(**common_kwargs)
            self.hands_right = self.mp_hands.Hands(**common_kwargs)
        except TypeError:
            # 혹시 아주 오래된/특이 버전이면 model_complexity 미지원일 수 있어요.
            # 그런 경우를 대비한 폴백.
            fallback_kwargs = dict(common_kwargs)
            fallback_kwargs.pop("model_complexity", None)
            self.hands_left = self.mp_hands.Hands(**fallback_kwargs)
            self.hands_right = self.mp_hands.Hands(**fallback_kwargs)

        self.mp_drawing = mp.solutions.drawing_utils


    def setup_display_windows(self):
        if self.cameras.get('left') is not None:
            cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL); cv2.resizeWindow('Left Camera',640,480); cv2.moveWindow('Left Camera',100,100)
        if self.cameras.get('right') is not None:
            cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL); cv2.resizeWindow('Right Camera',640,480); cv2.moveWindow('Right Camera',780,100)
        self.logger.info("Display windows initialized - press 'q' to quit")

    # --------- Servos ----------
    def setup_servo_defaults(self):
        robot=self.config.get('robot_arms',{})
        self.servo_ids=robot.get('motor_ids',[1,2,3,4])
        self.min_positions=[1024,1024,1024,1024]
        self.max_positions=[2944,3456,3200,3136]

    def setup_servos(self):
        if not DynamixelSDK_available: return
        robot=self.config.get('robot_arms',{}); follower=robot.get('follower',{})
        port=follower.get('port','/dev/follower_arm'); baud=follower.get('baudrate',1000000)
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(robot.get('protocol_version',2.0))
        if not self.port_handler.openPort(): raise Exception(f"Failed to open port {port}")
        if not self.port_handler.setBaudRate(baud): raise Exception(f"Failed to set baudrate {baud}")
        self.servo_ids=robot.get('motor_ids',[1,2,3,4])
        self.min_positions=[1280,1920,1120,1664]; self.max_positions=[3072,3072,3072,3072]
        goal_addr = robot.get('addr_goal_position',116)
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, goal_addr, 4)
        self.enable_torque()
        self.logger.info(f"Servos initialized on {port}")

    def enable_torque(self):
        if not DynamixelSDK_available: return
        addr=64
        for sid in self.servo_ids:
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, sid, addr, 1)
                if dxl_comm_result!=COMM_SUCCESS: self.logger.error(f"Failed to enable torque for servo {sid}")
                elif dxl_error!=0: self.logger.error(f"Servo {sid} error: {dxl_error}")
                else: self.logger.info(f"Torque enabled for servo {sid}")
            except Exception as e:
                self.logger.error(f"Exception enabling torque for servo {sid}: {e}")

    def disable_torque(self):
        if not DynamixelSDK_available: return
        addr=64
        for sid in self.servo_ids:
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, sid, addr, 0)
                if dxl_comm_result==COMM_SUCCESS and dxl_error==0:
                    self.logger.info(f"Torque disabled for servo {sid}")
            except Exception as e:
                self.logger.error(f"Exception disabling torque for servo {sid}: {e}")

    # ---- Present velocity poller (for movement start detection) ----
    def _any_servo_moving(self) -> bool:
        """Return True if any servo present velocity != 0."""
        if self.test_mode or (not DynamixelSDK_available): return False
        try:
            addr = self.DXL_ADDR_PRESENT_VELOCITY
            for sid in self.servo_ids:
                dxl_present_velocity, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, sid, addr
                )
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    if dxl_present_velocity != 0:
                        return True
            return False
        except Exception:
            return False

    def _wait_motion_start(self, deadline: float) -> Optional[float]:
        """Poll servos until motion starts or timeout; return timestamp or None."""
        if self.test_mode or (not DynamixelSDK_available): return None
        while time.time() < deadline:
            if self._any_servo_moving():
                return time.time()
            time.sleep(self.RT_POLL_INTERVAL)
        return None

    # ---------- Commands ----------
    def send_servo_commands(self, positions):
        if self.test_mode or not DynamixelSDK_available: return True
        if self.emergency_stop:
            self.logger.warning("Emergency stop active - not sending commands"); return False
        try:
            for i, pos in enumerate(positions):
                min_safe = self.safe_zone_min[i] if i < len(self.safe_zone_min) else 1200
                max_safe = self.safe_zone_max[i] if i < len(self.safe_zone_max) else 2896
                if pos < min_safe or pos > max_safe:
                    self.logger.error(f"Refusing unsafe position {pos} for joint {i}")
                    self.emergency_stop = True; return False
            self.group_sync_write.clearParam()
            for i, sid in enumerate(self.servo_ids):
                if i < len(positions):
                    p=int(positions[i])
                    pb=[DXL_LOBYTE(DXL_LOWORD(p)), DXL_HIBYTE(DXL_LOWORD(p)),
                        DXL_LOBYTE(DXL_HIWORD(p)), DXL_HIBYTE(DXL_HIWORD(p))]
                    self.group_sync_write.addParam(sid, pb)
            dxl_comm_result = self.group_sync_write.txPacket()
            success = dxl_comm_result == COMM_SUCCESS
            if success:
                self.last_successful_positions = positions.copy()
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
        if self.emergency_stop: return self.last_positions.copy()
        safe=[]
        for i, pos in enumerate(positions):
            mn=self.safe_zone_min[i]; mx=self.safe_zone_max[i]
            sp=max(mn, min(mx, int(pos)))
            if i < len(self.last_positions):
                lp=self.last_positions[i]
                sp=int(self.position_smoothing_alpha*sp + (1-self.position_smoothing_alpha)*lp)
            safe.append(sp)
        self.last_positions=safe.copy(); return safe

    # ---------- Cameras ----------
    def start_camera_threads(self):
        self.capture_active=True
        for name, cam in self.cameras.items():
            if cam is not None:
                self.frame_queues[name]=Queue(maxsize=2)
                t=threading.Thread(target=self._camera_capture_thread, args=(name,cam), daemon=True)
                t.start(); self.camera_threads[name]=t
    def _camera_capture_thread(self, name, cam):
        while self.capture_active:
            try:
                ret, frame = cam.read()
                if ret:
                    try: self.frame_queues[name].put_nowait(frame)
                    except:
                        try: self.frame_queues[name].get_nowait(); self.frame_queues[name].put_nowait(frame)
                        except Empty: pass
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Camera {name} error: {e}"); break
    def stop_camera_threads(self):
        self.capture_active=False
        for name, t in self.camera_threads.items(): t.join(timeout=1.0)
        self.camera_threads.clear()
    def get_latest_frames(self):
        frames={}
        for name in self.cameras.keys():
            try:
                frame=None
                while True:
                    try: frame=self.frame_queues[name].get_nowait()
                    except Empty: break
                frames[name]=frame
            except KeyError:
                frames[name]=None
        return frames

    def extract_hand_features(self, frame, camera_name):
        if frame is None: return [np.nan,np.nan], None
        try:
            if frame.size==0 or len(frame.shape)!=3:
                return [np.nan,np.nan], frame.copy()
            hp = self.hands_left if camera_name=='left' else self.hands_right
            frame.flags.writeable=False
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res=hp.process(rgb)
            frame.flags.writeable=True
            disp=frame.copy()
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(disp, lm, self.mp_hands.HAND_CONNECTIONS)
                tip=lm.landmark[8]
                if 0<=tip.x<=1 and 0<=tip.y<=1:
                    h,w=frame.shape[:2]; x=tip.x*w; y=tip.y*h
                    if 0<=x<=w and 0<=y<=h: return [float(x),float(y)], disp
            return [np.nan,np.nan], disp
        except Exception as e:
            self.logger.error(f"Hand detection error in {camera_name}: {e}")
            return [np.nan,np.nan], frame.copy()

    def predict_joint_positions(self, features):
        if self.model is None: return self.last_successful_positions.copy()
        try:
            X=np.array(features,dtype=np.float32).reshape(1,-1)
            if self.model_type=="xgb":
                y=self.model.predict(X); r=y[0]
                return r if np.isfinite(r).all() else self.last_successful_positions.copy()
            if not np.isfinite(X).all(): return self.last_successful_positions.copy()
            if self.normalize and self.scaler_X is not None: X=self.scaler_X.transform(X)
            Xt=torch.FloatTensor(X).to(self.device)
            with torch.no_grad(): pred=self.model(Xt).cpu().numpy()
            if self.normalize and self.scaler_y is not None: pred=self.scaler_y.inverse_transform(pred)
            r=pred[0]; return r if np.isfinite(r).all() else self.last_successful_positions.copy()
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.last_successful_positions.copy()

    # ---------- Display ----------
    def update_display(self, frames, lf, rf):
        if not self.show_display: return
        for name, frame in frames.items():
            if frame is not None:
                disp=frame.copy()
                cv2.rectangle(disp,(10,10),(300,110),(0,0,0),-1)
                if self.robot_stopped:
                    rs, rc = "STOPPED", (0,165,255)
                elif self.hand_tracking_enabled:
                    rs, rc = "TRACKING", (0,255,0)
                else:
                    rs, rc = "HOME", (255,255,0)
                cv2.putText(disp,f"ROBOT: {rs}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,rc,2)
                tc=(0,255,0) if self.hand_tracking_enabled else (0,0,255)
                cv2.putText(disp,f"TRACKING: {'ON' if self.hand_tracking_enabled else 'OFF'}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,tc,2)
                vc=(0,255,0) if self.voice_active else (0,0,255)
                cv2.putText(disp,f"VOICE: {'ACTIVE' if self.voice_active else 'INACTIVE'}",(20,70),cv2.FONT_HERSHEY_SIMPLEX,0.4,vc,2)
                ac=(0,255,0) if (self.arduino and self.arduino.is_open) else (0,0,255)
                cv2.putText(disp,f"ARDUINO: {'OK' if (self.arduino and self.arduino.is_open) else 'DISCONNECTED'}",(20,90),cv2.FONT_HERSHEY_SIMPLEX,0.4,ac,2)
                if self.hand_tracking_enabled:
                    x,y = (lf if name=='left' else rf)
                    if np.isfinite([x,y]).all():
                        color=(0,255,0) if name=='left' else (0,255,255)
                        cv2.circle(disp,(int(x),int(y)),12,color,-1)
                        cv2.circle(disp,(int(x),int(y)),15,(255,255,255),2)
                cv2.imshow(f"{name.title()} Camera", disp)
        if (cv2.waitKey(1) & 0xFF)==ord('q'): self.running=False

    # ---------- Voice thread (unchanged logic) ----------
    def voice_recognition_thread(self):
        print("🎤 Voice recognition thread started")
        consecutive_errors=0; max_errors=5
        def _proc_stream(desc, timeout, handler, for_command=False):
            stream_ctx=None
            try:
                stream_ctx, responses = start_stream(self.mic_index, self.mic_hint, self.debug, for_command)
            except Exception as e:
                self.logger.error(f"Failed to start {desc} stream: {e}")
                return "start_error"
            t0=time.time()
            try:
                for resp in responses:
                    if not self.running or not self.voice_active: return "stopped"
                    if timeout and (time.time()-t0)>timeout: return "timeout"
                    if resp.results and handler(resp): return "success"
                return "no_match"
            except Exception as e:
                self.logger.error(f"{desc} recognition error: {e}"); return "error"
            finally:
                if stream_ctx:
                    try: stream_ctx.__exit__(None,None,None)
                    except Exception as e: self.logger.error(f"Error closing {desc} stream: {e}")

        def wake_handler(resp)->bool:
            for r in resp.results:
                if r.alternatives and not r.is_final:
                    txt=r.alternatives[0].transcript.strip()
                    if txt and detect_wake_interim(txt): return True
            return False
        def cmd_handler(resp)->bool:
            for r in resp.results:
                if r.alternatives:
                    txt=r.alternatives[0].transcript.strip()
                    if txt:
                        cmd=detect_cmd_interim(txt)
                        if cmd: self.handle_voice_command(cmd); return True
            return False

        while self.running and self.voice_active:
            try:
                s=_proc_stream("wake",10.0,wake_handler,for_command=False)
                if s in ("start_error","error"): consecutive_errors+=1
                if s!="success":
                    if s=="stopped": break
                    if consecutive_errors>=max_errors:
                        print("⚠️ Too many voice recognition errors, stopping voice thread"); self.voice_active=False; break
                    time.sleep(0.5); continue
                print("✅ Wake word detected - Listening for command...")
                s2=_proc_stream("command",3.0,cmd_handler,for_command=False)
                if s2 in ("start_error","error"): consecutive_errors+=1
                else: consecutive_errors=0
            except Exception as e:
                self.logger.error(f"Voice recognition error: {e}")
                consecutive_errors+=1
                if consecutive_errors>=max_errors:
                    print("⚠️ Too many voice recognition errors, stopping voice thread"); self.voice_active=False; break
                time.sleep(0.5)
        print("🔇 Voice recognition thread stopped")

    def handle_voice_command(self, command):
        print(f"🗣️ Voice Command: {command}")
        if command=="EXIT": print("🛑 Exit command received"); self.running=False
        elif command=="STOP": self.robot_stopped=True; self.hand_tracking_enabled=False; print("⏹️ Robot STOPPED")
        elif command=="GO_HOME":
            self.robot_stopped=False; self.hand_tracking_enabled=False; print("🏠 Going HOME")
            if not self.test_mode:
                cp=self.clamp_positions(self.safe_holding_position); self.send_servo_commands(cp)
        elif command=="TRACKING_ON":
            self.robot_stopped=False; self.hand_tracking_enabled=True; print("📷 Hand tracking ENABLED")
        elif command in ("LED_ON","LED_OFF"):
            ar_cmd="ON" if command=="LED_ON" else "OFF"
            if self.arduino: send_arduino(self.arduino, ar_cmd)
            else: print(f"⚠️ Arduino not connected, cannot send {ar_cmd}")

    # ---------- Reaction-time helpers ----------
    def _rt_commit_and_maybe_report(self, cam_to_pred, cam_to_tx, cam_to_move, pred_to_tx):
        self.rt_cam_to_pred.append(cam_to_pred)
        self.rt_cam_to_tx.append(cam_to_tx)
        self.rt_pred_to_tx.append(pred_to_tx)
        # cam_to_move may be None if not measurable
        if cam_to_move is not None: self.rt_cam_to_move.append(cam_to_move)

        n = len(self.rt_cam_to_tx)  # same length for core metrics
        if n >= self.RT_BATCH:
            def _summ(a: List[float]):
                if not a: return (float('nan'), float('nan'), float('nan'))
                return (float(np.mean(a)), float(np.min(a)), float(np.max(a)))
            m1=_summ(self.rt_cam_to_pred)
            m2=_summ(self.rt_pred_to_tx)
            m3=_summ(self.rt_cam_to_tx)
            m4=_summ(self.rt_cam_to_move)

            print(f"[Reaction Time] (n={n}) "
                  f"cam→pred avg={m1[0]*1e3:,.2f} ms | min={m1[1]*1e3:,.2f} | max={m1[2]*1e3:,.2f}   "
                  f"pred→tx avg={m2[0]*1e3:,.2f} ms | min={m2[1]*1e3:,.2f} | max={m2[2]*1e3:,.2f}   "
                  f"cam→tx avg={m3[0]*1e3:,.2f} ms | min={m3[1]*1e3:,.2f} | max={m3[2]*1e3:,.2f}")
            if len(self.rt_cam_to_move) > 0:
                print(f"[Reaction Time] cam→move avg={m4[0]*1e3:,.2f} ms | min={m4[1]*1e3:,.2f} | max={m4[2]*1e3:,.2f} (measured)")
            else:
                print(f"[Reaction Time] cam→move: N/A (no motor feedback)")

            # reset buffers
            self.rt_cam_to_pred.clear(); self.rt_pred_to_tx.clear()
            self.rt_cam_to_tx.clear(); self.rt_cam_to_move.clear()

    # ---------- Main loop ----------
    def run_concurrent_loop(self):
        print("🤖 Starting system:")
        print(f"   📷 Hand tracking: {'ENABLED' if self.hand_tracking_enabled else 'DISABLED'}")
        print(f"   🎤 Voice control: {'ENABLED' if self.voice_active else 'DISABLED'}")

        self.start_camera_threads()
        if self.voice_active:
            self.voice_thread = threading.Thread(target=self.voice_recognition_thread, daemon=True)
            self.voice_thread.start()

        try:
            while self.running:
                loop_start = time.time()

                # === t_cap: 바로 프레임을 끌어온 시점 ===
                frames = self.get_latest_frames()
                t_cap = time.time()

                left_features, left_processed = self.extract_hand_features(frames.get('left'), 'left')
                right_features, right_processed = self.extract_hand_features(frames.get('right'), 'right')
                t_hand = time.time()

                if left_processed is not None: frames['left']=left_processed
                if right_processed is not None: frames['right']=right_processed
                if self.show_display: self.update_display(frames, left_features, right_features)

                # 유효 프레임(손 인식 성공) 판단: 좌/우 모두 (x,y) 유효해야 모델 입력 4개가 모두 유효
                left_ok  = np.isfinite(left_features).all()
                right_ok = np.isfinite(right_features).all()
                valid_hand = left_ok and right_ok

                if self.robot_stopped:
                    # 정지 상태: 측정하지 않음
                    pass

                elif self.hand_tracking_enabled and self.model is not None and valid_hand:
                    # === 예측 ===
                    combined = left_features + right_features
                    preds = self.predict_joint_positions(combined)
                    t_pred = time.time()

                    # === 전송 ===
                    if not self.test_mode:
                        clamped = self.clamp_positions(preds)
                        # 서보 움직임 감지 준비: 보내기 직전 데드라인 계산
                        move_deadline = time.time() + self.RT_MOVE_TIMEOUT
                        send_ok = self.send_servo_commands(clamped)
                        t_tx = time.time()

                        # 관측 ONLY: 움직임 시작 감지(옵션)
                        t_move = self._wait_motion_start(move_deadline) if send_ok else None

                        # 지표 기록 & 요약
                        cam_to_pred = (t_pred - t_cap)
                        pred_to_tx  = (t_tx   - t_pred)
                        cam_to_tx   = (t_tx   - t_cap)
                        cam_to_move = (t_move - t_cap) if t_move is not None else None
                        self._rt_commit_and_maybe_report(cam_to_pred, cam_to_tx, cam_to_move, pred_to_tx)
                    else:
                        # 테스트 모드: 전송 안 함. cam→tx/→move는 측정 불가
                        t_pred = time.time()
                        cam_to_pred = (t_pred - t_cap)
                        self._rt_commit_and_maybe_report(cam_to_pred, float('nan'), None, float('nan'))

                else:
                    # 추적 OFF 또는 손 미검출 또는 모델 없음 → 안전 자세 유지 (측정 X)
                    if not self.test_mode:
                        clamped = self.clamp_positions(self.safe_holding_position)
                        self.send_servo_commands(clamped)

                # 프레임 레이트 조절
                elapsed = time.time() - loop_start
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user")
        finally:
            self.cleanup()

    # ---------- Cleanup ----------
    def cleanup(self):
        self.logger.info("Cleaning up resources...")
        self.running=False
        if not self.test_mode and DynamixelSDK_available:
            self.logger.info("Moving servos to home position...")
            try:
                home=[2048,3328,1140,3072]
                cp=self.clamp_positions(home)
                if self.send_servo_commands(cp):
                    time.sleep(1.0); self.logger.info("Servos moved to home position")
                else:
                    self.logger.warning("Failed to move servos to home position")
                self.disable_torque(); self.logger.info("Servo torque disabled")
            except Exception as e:
                self.logger.error(f"Error during servo cleanup: {e}")

        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2.0)

        self.stop_camera_threads()
        for cam in self.cameras.values():
            if cam is not None:
                try: cam.release()
                except Exception as e: self.logger.error(f"Failed to release camera: {e}")
        if self.show_display: cv2.destroyAllWindows()

        if self.arduino and self.arduino.is_open:
            try:
                self.logger.info("Turning off Arduino LED...")
                send_arduino(self.arduino, "OFF"); time.sleep(0.5)
                self.logger.info("Arduino LED turned off")
            except Exception as e:
                self.logger.error(f"Failed to turn off Arduino LED: {e}")
            self.arduino.close()

        if not self.test_mode and DynamixelSDK_available and hasattr(self,'port_handler'):
            try: self.port_handler.closePort(); self.logger.info("Servo port closed")
            except Exception as e: self.logger.error(f"Error closing servo port: {e}")
        self.logger.info("Cleanup completed")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Concurrent Voice + Hardware Runner (legacy + RT)")
    parser.add_argument("--model", help="Path to trained model (.joblib or .pth)")
    parser.add_argument("--config", default="hardware_config.json")
    parser.add_argument("--arduino-port", default="/dev/arduino")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--display", action='store_true')
    parser.add_argument("--no-camera", action='store_true')
    parser.add_argument("--no-voice", action='store_true')
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--mic-index", type=int)
    parser.add_argument("--mic-hint", default=DEFAULT_MIC_HINT)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    if args.list_mics:
        list_input_devices(); return

    print("🤖 Hand Tracking + Voice Control System")
    print("="*50)
    print("Main: Camera hand tracking → Robot arm control")
    print("Voice: Wake '하이봇' → commands ('추적 시작', '정지', '홈', '켜/꺼', '그만')")
    print("Default: Tracking DISABLED | 'q' to quit")
    print("="*50)

    try:
        runner = ConcurrentVoiceHardwareRunner(
            model_path=args.model, hardware_config_path=args.config,
            arduino_port=args.arduino_port, test_mode=args.test,
            target_fps=args.fps, show_display=args.display,
            mic_index=args.mic_index, mic_hint=args.mic_hint, debug=args.debug
        )
        runner.camera_active = not args.no_camera
        runner.voice_active  = not args.no_voice
        if not runner.camera_active and not runner.voice_active:
            print("❌ Both camera and voice disabled. Nothing to run."); return
        runner.run_concurrent_loop()
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
