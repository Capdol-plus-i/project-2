import os
import sys
import time
import re
import unicodedata
import audioop
from six.moves import queue

import pyaudio
import webrtcvad
from google.cloud import speech

# ---- gRPC 경고 억제(선택) ----
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_LOG_SEVERITY_LEVEL", "ERROR")

# =======================
# 기본 설정(엔진은 16 kHz 고정)
# =======================
RATE = 16000               # webrtcvad 허용: 8000/16000/32000
FRAME_MS = 20              # 10/20/30만 허용
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)   # 320
BYTES_PER_SAMPLE = 2       # S16LE
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE

WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이"]
CMD_MAP = {
    "왼쪽": ["왼쪽", "왼 쪽", "왼"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른", "5"],
    "위": ["위", "위로", "위쪽"],
    "아래": ["아래", "아래로", "아레"],
    "종료": ["종료", "끝내", "종료해"]
}

# 장치 선택 힌트(환경변수로 덮어쓰기 가능)
MIC_HINT = os.environ.get("MIC_HINT", "Blue Tiki")  # "Mic", "USB" 등도 OK

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", "", text)

def pick_input_device_index(p: pyaudio.PyAudio, hint: str | None = None) -> int | None:
    hint = (hint or "").lower()
    chosen = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        name = f'{info.get("name","")}'.lower()
        host = f'{p.get_host_api_info_by_index(info.get("hostApi",0)).get("name","")}'.lower()
        if hint and (hint in name or hint in host):
            return i
        if chosen is None:
            chosen = i
    return chosen

class MicrophoneStream:
    """
    하드웨어 샘플레이트(hw_rate)로 마이크를 열고, audioop.ratecv로 16 kHz/20 ms 프레임을 만들어
    webrtcvad로 음성만 내보낸다.
    """
    def __init__(self, target_rate, frame_ms, device_index=None):
        self._target_rate = target_rate  # 16000
        self._frame_ms = frame_ms        # 20
        self._target_bytes_per_frame = int(target_rate * frame_ms / 1000) * BYTES_PER_SAMPLE

        self._buff = queue.Queue()
        self._carry = b""
        self._ratecv_state = None
        self.vad = webrtcvad.Vad(2)

        self._device_index = device_index
        self._hw_rate = None
        self.closed = True

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        if self._device_index is None:
            self._device_index = pick_input_device_index(self._pa, MIC_HINT)
        if self._device_index is None:
            raise RuntimeError("입력 장치를 찾을 수 없습니다. MIC_HINT 환경변수로 힌트를 주세요.")

        # 장치의 기본 샘플레이트로 오픈 (호환성 ↑)
        dinfo = self._pa.get_device_info_by_index(self._device_index)
        hw_rate = int(round(dinfo.get("defaultSampleRate", 16000)))
        # 그래도 안 맞을 수 있으니 실패 시 몇 개 후보로 재시도
        candidates = [hw_rate, 48000, 44100, 32000, 16000]
        last_err = None
        for r in candidates:
            try:
                frames_per_buffer = int(r * self._frame_ms / 1000)
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=r,
                    input=True,
                    input_device_index=self._device_index,
                    frames_per_buffer=frames_per_buffer,
                    stream_callback=self._fill_buffer,
                )
                self._hw_rate = r
                break
            except Exception as e:
                last_err = e
        if self._hw_rate is None:
            raise RuntimeError(f"마이크 열기 실패: {last_err}")

        self.closed = False
        print(f"🎤 Mic opened @ {self._hw_rate} Hz (device: {dinfo.get('name')})")
        return self

    def __exit__(self, exc_type, value, traceback):
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self.closed = True
            self._buff.put(None)
            self._pa.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def _to_target_rate(self, data: bytes) -> bytes:
        """hw_rate → target_rate(=16k) 변환"""
        if self._hw_rate == self._target_rate:
            return data
        converted, self._ratecv_state = audioop.ratecv(
            data,             # raw bytes
            BYTES_PER_SAMPLE, # width=2 (S16LE)
            1,                # nchannels
            self._hw_rate,    # inrate
            self._target_rate,# outrate
            self._ratecv_state
        )
        return converted

    def generator(self):
        try:
            while not self.closed:
                chunk = self._buff.get()
                if chunk is None:
                    return
                # 1) hw_rate → 16k로 리샘플
                pcm16k = self._to_target_rate(chunk)
                # 2) 20ms 프레임으로 쪼개기
                self._carry += pcm16k
                while len(self._carry) >= self._target_bytes_per_frame:
                    frame = self._carry[:self._target_bytes_per_frame]
                    self._carry = self._carry[self._target_bytes_per_frame:]
                    # 3) VAD
                    if self.vad.is_speech(frame, RATE):
                        yield frame
        except GeneratorExit:
            return

def build_client_and_config(is_command_mode=False):
    client = speech.SpeechClient()

    phrases = list(WAKE_WORDS)
    for v in CMD_MAP.values():
        phrases.extend(v)

    speech_context = speech.SpeechContext(phrases=phrases, boost=15.0)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,  # 16k
        language_code="ko-KR",
        speech_contexts=[speech_context],
        model="command_and_search",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=not is_command_mode,
        single_utterance=is_command_mode
    )
    return client, streaming_config

def start_stream(is_command_mode=False, device_index=None):
    client, streaming_config = build_client_and_config(is_command_mode=is_command_mode)
    mic = MicrophoneStream(RATE, FRAME_MS, device_index=device_index)
    mic.__enter__()
    audio_generator = mic.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=frame) for frame in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)
    return mic, responses

def main():
    print("🎙️ 준비 완료. 웨이크워드를 말하세요 (예: '하이봇')")
    try:
        while True:
            stream, responses = start_stream(is_command_mode=False)
            for response in responses:
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                transcript = normalize(result.alternatives[0].transcript.strip())
                if transcript:
                    print(" 인식:", transcript)

                if any(normalize(w) in transcript for w in WAKE_WORDS):
                    if result.is_final or result.stability > 0.8:
                        print(" ✅ 웨이크워드 감지 → 명령을 말씀하세요.")
                        stream.__exit__(None, None, None)
                        time.sleep(0.05)

                        cmd_stream, cmd_responses = start_stream(is_command_mode=True)
                        start_time = time.time()
                        MAX_COMMAND_DURATION = 3.0

                        for cmd_response in cmd_responses:
                            if time.time() - start_time > MAX_COMMAND_DURATION:
                                print(" ⏰ 명령 대기 초과")
                                cmd_stream.__exit__(None, None, None)
                                break

                            if not cmd_response.results:
                                continue
                            cmd_result = cmd_response.results[0]
                            if not cmd_result.alternatives:
                                continue
                            cmd_transcript = normalize(cmd_result.alternatives[0].transcript.strip())
                            if cmd_transcript:
                                print(" 명령:", cmd_transcript)

                            matched = False
                            for cmd, variations in CMD_MAP.items():
                                if any(normalize(v) in cmd_transcript for v in variations):
                                    matched = True
                                    print(f" 👉 명령어: {cmd}")
                                    if cmd == "종료":
                                        print(" 시스템 종료")
                                        cmd_stream.__exit__(None, None, None)
                                        return
                                    cmd_stream.__exit__(None, None, None)
                                    break

                            if not matched:
                                print(" ❓ 인식 실패. 다시 말씀해주세요.")
                            break
                        break
    except KeyboardInterrupt:
        print("\n^C 종료")
    except Exception as e:
        print("에러:", e, file=sys.stderr)

if __name__ == "__main__":
    main()