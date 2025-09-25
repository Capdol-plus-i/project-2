#!/usr/bin/env python3
import os
import time
import re
import unicodedata
import queue
import audioop
import pyaudio
import webrtcvad
from google.cloud import speech

# ---- 불필요한 로그 억제 ----
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"

# =======================
# 기본 설정
# =======================
TARGET_RATE = 16000
FRAME_MS = 20
BYTES_PER_SAMPLE = 2
SAMPLES_PER_FRAME = int(TARGET_RATE * FRAME_MS / 1000)
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE

MIC_HINT = "blue tiki"   # 🎤 Blue Tiki USB Mic 고정

WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이","5"]
CMD_MAP = {
    "왼쪽": ["왼쪽", "왼 쪽", "왼"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른", "5"],
    "위": ["위", "위로", "위쪽"],
    "아래": ["아래", "아래로", "아레","하이"],
    "종료": ["종료", "끝내", "종료해"]
}

def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))

def pick_blue_tiki_index(p: pyaudio.PyAudio) -> int:
    """Blue Tiki USB Mic만 선택"""
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            name = info.get("name", "").lower()
            if MIC_HINT in name:
                return i
    raise RuntimeError("Blue Tiki USB Mic 장치를 찾을 수 없습니다.")

class MicrophoneStream:
    def __init__(self, use_vad: bool):
        self.use_vad = use_vad
        self._pa = None
        self._stream = None
        self._buff = queue.Queue(maxsize=100)
        self._carry = b""
        self._ratecv_state = None
        self._hw_rate = None
        self._hw_channels = 1
        self.closed = True
        self.vad = webrtcvad.Vad(1) if use_vad else None

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        device_index = pick_blue_tiki_index(self._pa)
        dinfo = self._pa.get_device_info_by_index(device_index)
        default_rate = int(dinfo.get("defaultSampleRate", 48000))
        candidates = [default_rate, 48000, 44100, 32000, 16000]

        last_err = None
        for ch in (1, 2):
            for r in candidates:
                try:
                    frames_per_buffer = int(r * FRAME_MS / 1000)
                    self._stream = self._pa.open(
                        format=pyaudio.paInt16,
                        channels=ch,
                        rate=r,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=frames_per_buffer,
                        stream_callback=self._fill_buffer,
                    )
                    self._hw_rate = r
                    self._hw_channels = ch
                    self.closed = False
                    print(f"🎤 Blue Tiki Mic opened @ {self._hw_rate} Hz, ch={self._hw_channels}")
                    return self
                except Exception as e:
                    last_err = e
        raise RuntimeError(f"Blue Tiki 마이크 열기 실패: {last_err}")

    def __exit__(self, exc_type, value, traceback):
        self.closed = True
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except:
                pass
        try:
            self._buff.put_nowait(None)
        except:
            pass

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        try:
            if self._buff.full():
                self._buff.get_nowait()
            self._buff.put_nowait(in_data)
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def _to_mono_16k(self, data: bytes) -> bytes:
        pcm = data
        if self._hw_channels == 2:
            pcm = audioop.tomono(pcm, BYTES_PER_SAMPLE, 0.5, 0.5)
        if self._hw_rate != TARGET_RATE:
            pcm, self._ratecv_state = audioop.ratecv(
                pcm, BYTES_PER_SAMPLE, 1,
                self._hw_rate, TARGET_RATE,
                self._ratecv_state
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
            self._carry += pcm16k
            while len(self._carry) >= FRAME_BYTES:
                frame = self._carry[:FRAME_BYTES]
                self._carry = self._carry[FRAME_BYTES:]
                if self.vad:
                    if self.vad.is_speech(frame, TARGET_RATE):
                        yield frame
                else:
                    yield frame

def build_client_and_config():
    client = speech.SpeechClient()
    phrases = list(WAKE_WORDS)
    for v in CMD_MAP.values():
        phrases.extend(v)
    speech_context = speech.SpeechContext(phrases=phrases, boost=15.0)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_RATE,
        language_code="ko-KR",
        speech_contexts=[speech_context],
        model="command_and_search",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False  # ✅ 항상 열어두기
    )
    return client, streaming_config

def start_stream(is_command_mode=False):
    client, streaming_config = build_client_and_config()
    stream = MicrophoneStream(use_vad=not is_command_mode)
    stream.__enter__()
    audio_gen = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=f) for f in audio_gen)
    responses = client.streaming_recognize(streaming_config, requests)
    return stream, responses

def listen_for_wake_word(max_session_sec=55):
    start_ts = time.time()
    stream, responses = start_stream(is_command_mode=False)
    try:
        for response in responses:
            if time.time() - start_ts > max_session_sec:
                stream.__exit__(None, None, None)
                return False
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = normalize(result.alternatives[0].transcript.strip())
            if not transcript:
                continue
            print(" 인식:", transcript)
            if any(normalize(w) in transcript for w in WAKE_WORDS):
                if result.is_final or getattr(result, "stability", 0.0) > 0.5:
                    stream.__exit__(None, None, None)
                    return True
    except Exception:
        try:
            stream.__exit__(None, None, None)
        except:
            pass
        return False
    try:
        stream.__exit__(None, None, None)
    except:
        pass
    return False

def main():
    print("🎙️ 준비 완료. 웨이크워드를 말해주세요 (예: '하이봇')")
    while True:
        got_wake = listen_for_wake_word()
        if not got_wake:
            continue
        print("✅ 웨이크워드 감지됨 → 명령을 말씀하세요.")
        time.sleep(0.3)
        cmd_stream, cmd_responses = start_stream(is_command_mode=True)
        start_time = time.time()
        TIMEOUT = 3.0
        for cmd_response in cmd_responses:
            if time.time() - start_time > TIMEOUT:
                print("⏰ 명령어 대기 시간 초과")
                cmd_stream.__exit__(None, None, None)
                break
            if not cmd_response.results:
                continue
            cmd_result = cmd_response.results[0]
            if not cmd_result.alternatives:
                continue
            cmd_text = normalize(cmd_result.alternatives[0].transcript.strip())
            if not cmd_text:
                continue
            print(" 명령 인식:", cmd_text)
            matched = False
            for cmd, variations in CMD_MAP.items():
                if any(normalize(v) in cmd_text for v in variations):
                    matched = True
                    print(f"👉 명령어: {cmd}")
                    if cmd == "종료":
                        print("🛑 시스템 종료")
                        cmd_stream.__exit__(None, None, None)
                        return
                    cmd_stream.__exit__(None, None, None)
                    break
            if not matched:
                print("❓ 인식 실패. 다시 말씀해주세요.")
                cmd_stream.__exit__(None, None, None)
            break
        time.sleep(0.1)

if __name__ == "__main__":
    main()
