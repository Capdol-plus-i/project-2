#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blue Tiki USB Mic + Google Speech → Arduino (음성명령 전용, 저지연 튜닝)
- 프레임 10ms, 연속 스트리밍(프레임 드랍 없음)
- 웨이크워드: interim '정확 포함' 빠른 트리거 + final 안전망
- 명령: interim '정확 포함' 빠른 트리거(OFF 우선) + final 안전망
- 명령 구간은 single_utterance=True 로 final 빨리 받기
- 16k/mono 안전 변환(스테레오 평균, ratecv 상태 유지)
- 마이크 목록/선택 (--list-mics, --mic-index, --mic-hint)
- 디버깅(--debug): interim/final 로그, 간단 VAD 모니터
"""

import os
import time
import re
import unicodedata
import queue
import audioop
import ctypes
import ctypes.util
import struct
import argparse
from typing import Optional, List, Tuple

import serial
import pyaudio
import webrtcvad
from google.cloud import speech

# ---------- 로그 억제 ----------
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

# ---------- 기본 설정 ----------
TARGET_RATE = 16000
FRAME_MS = 10  # 저지연: 10ms
BYTES_PER_SAMPLE = 2
SAMPLES_PER_FRAME = int(TARGET_RATE * FRAME_MS / 1000)
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE

DEFAULT_MIC_HINT = "blue"  # "blue tiki" 대신 "blue"/"usb" 등으로도 매치 가능

# 웨이크워드
WAKE_CANONICAL = "하이봇"
WAKE_VARIANTS = [
    "하이봇", "하이 봇", "하이봇아", "하 이 봇",
    "아이봇", "하이보", "하이 보트"  # 흔한 오인식들도 추가
]

# 명령 사전 (OFF 우선)
COMMAND_SYNONYMS = {
    "OFF": ["꺼", "꺼줘", "불꺼", "불 꺼", "라이트오프", "라이트 오프", "끄자"],
    "ON":  ["켜", "켜줘", "불켜", "불 켜", "라이트온", "라이트 온", "키자"],
    "EXIT": ["종료", "끝내", "그만", "나가", "종 료"]
}

# ---------- 유틸 ----------
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
    # 짧은(<=2) 키워드: 정확 포함만
    for cmd, syns in COMMAND_SYNONYMS.items():
        for k in syns:
            kn = normalize(k)
            if len(kn) <= 2 and kn in s:
                return cmd
    # 3글자 이상: 정확 포함
    for cmd, syns in COMMAND_SYNONYMS.items():
        for k in syns:
            kn = normalize(k)
            if len(kn) >= 3 and kn in s:
                return cmd
    # 3글자 이상: 퍼지(≤1)
    for cmd, syns in COMMAND_SYNONYMS.items():
        for k in syns:
            kn = normalize(k)
            if len(kn) >= 3 and fuzzy_match_word(s, kn, 1):
                return cmd
    return None

# ---- 저지연용: interim 빠른 트리거(정확 포함만) ----
def quick_contains(text: str, keys: List[str]) -> bool:
    s = normalize(text)
    return any(normalize(k) in s for k in keys)

def detect_wake_interim(text: str) -> bool:
    keys = [WAKE_CANONICAL] + WAKE_VARIANTS
    return quick_contains(text, keys)

def detect_cmd_interim(text: str) -> Optional[str]:
    # OFF → ON → EXIT 순(OFF 우선)
    if quick_contains(text, COMMAND_SYNONYMS["OFF"]): return "OFF"
    if quick_contains(text, COMMAND_SYNONYMS["ON"]):  return "ON"
    if quick_contains(text, COMMAND_SYNONYMS["EXIT"]):return "EXIT"
    return None

# ---------- 마이크 ----------
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
    """
    - 연속 오디오 스트리밍(프레임 드랍 X)
    - 16k/mono 변환은 안전하게 처리
    - VAD는 디버그 모니터링용(결정/드랍에는 사용 안 함)
    """
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
                    frames_per_buffer = int(r * FRAME_MS / 1000)  # 10ms
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

            yield pcm16k  # 드랍 없이 연속 스트리밍

# ---------- Google Speech ----------
def build_client_and_config(single_utter: bool):
    client = speech.SpeechClient()

    phrases = list(set(
        [WAKE_CANONICAL] + WAKE_VARIANTS +
        [
            "불 꺼 줘", "불켜 줘", "라이트 오프", "라이트 온",
            "라이트오프", "라이트온", "종료해", "끝내", "그만"
        ] + sum(COMMAND_SYNONYMS.values(), [])
    ))
    speech_context = speech.SpeechContext(phrases=phrases, boost=20.0)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_RATE,
        language_code="ko-KR",
        speech_contexts=[speech_context],
        max_alternatives=3,
        enable_automatic_punctuation=False,
        use_enhanced=True,              # 미지원이면 무시됨
        model="command_and_search",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,           # interim 빠른 트리거 용
        single_utterance=single_utter   # 명령 구간만 True
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

# ---------- Arduino ----------
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

# ---------- 텍스트 판정 ----------
def any_final_texts(response) -> List[str]:
    texts: List[str] = []
    if not response.results:
        return texts
    for res in response.results:
        if not res.alternatives:
            continue
        if res.is_final:
            for alt in res.alternatives:
                t = alt.transcript.strip()
                if t:
                    texts.append(t)
    return texts

def detect_wake_from_texts(texts: List[str], debug: bool=False) -> bool:
    for t in texts:
        if debug: print(f"[FINAL/WAKE] {t}")
        if is_wake_word(t):
            return True
    return False

def detect_cmd_from_texts(texts: List[str], debug: bool=False) -> Tuple[Optional[str], Optional[str]]:
    for t in texts:
        if debug: print(f"[FINAL/CMD] {t}")
        cmd = which_command(t)
        if cmd:
            return cmd, t
    return None, None

# ---------- 메인 루프 ----------
def run_voice_mode(port: str, mic_index: Optional[int], mic_hint: str, debug: bool):
    ser = open_arduino(port)
    print("🎙️ 준비 완료. 웨이크워드를 말해주세요 (예: '하이봇')")

    try:
        while True:
            # ---- 웨이크워드 모드 ----
            wake_stream, wake_responses = start_stream(mic_index, mic_hint, debug, for_command=False)
            wake_start, WAKE_SESSION_MAX = time.time(), 55.0
            got_wake = False
            try:
                for response in wake_responses:
                    if time.time() - wake_start > WAKE_SESSION_MAX:
                        break

                    # 1) interim 빠른 트리거 (정확 포함만)
                    if response.results:
                        fired = False
                        for res in response.results:
                            if res.alternatives and not res.is_final:
                                t = res.alternatives[0].transcript.strip()
                                if t and detect_wake_interim(t):
                                    if debug: print(f"[INTERIM/WAKE] {t}")
                                    got_wake = True
                                    fired = True
                                    break
                        if fired:
                            break

                    # 2) final 안전 트리거
                    finals = any_final_texts(response)
                    if finals and detect_wake_from_texts(finals, debug):
                        got_wake = True
                        break
            finally:
                wake_stream.__exit__(None, None, None)

            if not got_wake:
                continue

            print("✅ 웨이크워드 감지 → 명령을 말씀하세요.")

            # ---- 명령 모드 (single_utterance=True 로 final 빠르게) ----
            cmd_stream, cmd_responses = start_stream(mic_index, mic_hint, debug, for_command=True)
            CMD_TIMEOUT, cmd_start = 4.0, time.time()
            try:
                fast_fired = False
                for response in cmd_responses:
                    if time.time() - cmd_start > CMD_TIMEOUT:
                        print("⏰ 명령어 대기 시간 초과 (웨이크워드로 복귀)")
                        break

                    # 1) interim 빠른 트리거 (정확 포함만, OFF 우선)
                    if response.results and not fast_fired:
                        for res in response.results:
                            if res.alternatives and not res.is_final:
                                t = res.alternatives[0].transcript.strip()
                                if t:
                                    cmd = detect_cmd_interim(t)
                                    if cmd:
                                        if debug: print(f"[INTERIM/CMD] {t} -> {cmd}")
                                        if cmd == "EXIT":
                                            print("🛑 시스템 종료")
                                            return
                                        if ser: send_arduino(ser, cmd)
                                        fast_fired = True
                                        break
                        if fast_fired:
                            break

                    # 2) final 안전 트리거
                    finals = any_final_texts(response)
                    if finals:
                        cmd, raw = detect_cmd_from_texts(finals, debug)
                        if cmd:
                            if cmd == "EXIT":
                                print("🛑 시스템 종료")
                                return
                            if ser: send_arduino(ser, cmd)
                            break
            finally:
                cmd_stream.__exit__(None, None, None)

            time.sleep(0.1)

    finally:
        if ser and ser.is_open:
            ser.close()

# ---------- 실행 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/arduino", help="Arduino 포트 (기본: /dev/arduino)")
    parser.add_argument("--list-mics", action="store_true", help="입력 장치 목록 표시 후 종료")
    parser.add_argument("--mic-index", type=int, default=None, help="사용할 입력 장치 인덱스")
    parser.add_argument("--mic-hint", default=DEFAULT_MIC_HINT, help=f"입력 장치 이름 힌트(기본: '{DEFAULT_MIC_HINT}')")
    parser.add_argument("--debug", action="store_true", help="중간/최종 인식 로그 표시")
    args = parser.parse_args()

    if args.list_mics:
        list_input_devices()
        return

    run_voice_mode(args.port, args.mic_index, args.mic_hint, args.debug)

if __name__ == "__main__":
    main()
