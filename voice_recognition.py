import os
import time
import math
import re
import unicodedata
from collections import deque
from six.moves import queue

import pyaudio
import webrtcvad
from google.cloud import speech

# gRPC 로그 억제(선택)
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_LOG_SEVERITY_LEVEL", "ERROR")

# =======================
# 기본 설정
# =======================
RATE = 16000               # Google STT 권장 16k
FRAME_MS = 20              # VAD 허용값: 10/20/30ms → 20ms 권장
SAMPLES_PER_FRAME = RATE * FRAME_MS // 1000     # 320 samples
BYTES_PER_SAMPLE = 2                           # 16-bit PCM
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE  # 640 bytes

# VAD padding: 앞 200ms, 뒤 400ms 정도
PAD_LEADING_MS = 200
PAD_TRAILING_MS = 400
PAD_LEADING_FRAMES = PAD_LEADING_MS // FRAME_MS
PAD_TRAILING_FRAMES = PAD_TRAILING_MS // FRAME_MS

# 장치 선택 키워드(마이크 이름 일부). 비워두면 기본 입력 사용
MIC_DEVICE_KEYWORD = os.getenv("MIC_DEVICE_KEYWORD", "")  # 예: "Blue Tiki" / "USB"

WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이"]
CMD_MAP = {
    "왼쪽":   ["왼쪽", "왼 쪽", "왼"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른", "5"],
    "위":     ["위", "위로", "위쪽"],
    "아래":   ["아래", "아래로", "아레"],
    "종료":   ["종료", "끝내", "종료해"],
}

def normalize(text: str) -> str:
    """한글 인식용: 유니코드 정규화 + 공백 제거"""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text)

# =======================
# 마이크 스트림 (PyAudio + webrtcvad)
# =======================
class MicrophoneStream:
    """PyAudio 콜백으로 raw PCM 수집 → 20ms 프레임화 → VAD로 필터.
       무음 구간에선 프레임을 버리고, 발화 경계는 앞/뒤 padding을 붙여준다.
    """
    def __init__(self, rate: int, frame_ms: int, device_keyword: str = ""):
        self.rate = rate
        self.chunk = rate * frame_ms // 1000
        self._buff = queue.Queue()
        self.closed = True
        self.vad = webrtcvad.Vad(2)  # 0 관대 ~ 3 엄격
        self._carry = b""
        self.device_keyword = device_keyword
        self._leading = deque(maxlen=PAD_LEADING_FRAMES)
        self._trailing_count = 0
        self._in_speech = False

    # ---- PyAudio helpers ----
    def _find_input_device_index(self, pa: pyaudio.PyAudio):
        if not self.device_keyword:
            return None
        kw = self.device_keyword.lower()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if info.get("maxInputChannels", 0) > 0 and kw in name:
                return i
        return None

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        idx = self._find_input_device_index(self._pa)
        if idx is not None:
            print(f"🎤 Using input device #{idx} matching '{self.device_keyword}'")
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=idx,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if hasattr(self, "_stream"):
                self._stream.stop_stream()
                self._stream.close()
        finally:
            self.closed = True
            self._buff.put(None)
            if hasattr(self, "_pa"):
                self._pa.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # 콜백에서 들어온 바이트를 그대로 큐에 적재
        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def _frames(self):
        """20ms 바이트 프레임 생성기 (무가공)"""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            self._carry += chunk
            while len(self._carry) >= FRAME_BYTES:
                frame = self._carry[:FRAME_BYTES]
                self._carry = self._carry[FRAME_BYTES:]
                yield frame

    def vad_frames(self):
        """VAD로 말 구간만 내보내되, 앞뒤 padding 포함."""
        for frame in self._frames():
            is_speech = self.vad.is_speech(frame, self.rate)

            if is_speech:
                if not self._in_speech:
                    # 새 발화 시작: 앞 padding 뱉기
                    for f in self._leading:
                        yield f
                    self._in_speech = True
                    self._trailing_count = 0
                yield frame
            else:
                if self._in_speech:
                    # 말하던 중 잠깐의 무음: trailing 카운트
                    self._trailing_count += 1
                    yield frame  # trailing 동안도 붙여서 자연스럽게
                    if self._trailing_count >= PAD_TRAILING_FRAMES:
                        # 발화 종료로 간주 → 상태 초기화
                        self._in_speech = False
                        self._trailing_count = 0
                        self._leading.clear()
                else:
                    # 무음 대기: leading 버퍼 축적
                    self._leading.append(frame)
                    # 무음 자체는 바깥으로 안 보냄

# =======================
# Google Speech Stream
# =======================
def start_stream(is_command_mode=False):
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.")
    client = speech.SpeechClient()

    # 힌트/부스트
    phrases = WAKE_WORDS + [v for vs in CMD_MAP.values() for v in vs]
    speech_context = speech.SpeechContext(phrases=phrases, boost=15.0)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
        speech_contexts=[speech_context],
        model="command_and_search",
        enable_automatic_punctuation=False,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=not is_command_mode,   # 웨이크 단계: interim 허용
        single_utterance=is_command_mode       # 명령 단계: 한 문장 인식 후 종료
    )

    mic = MicrophoneStream(RATE, FRAME_MS, device_keyword=MIC_DEVICE_KEYWORD)

    def request_generator():
        with mic as stream:
            for frame in stream.vad_frames():
                yield speech.StreamingRecognizeRequest(audio_content=frame)

    # responses는 제너레이터(스트리밍)
    responses = client.streaming_recognize(streaming_config, request_generator())
    return responses  # 마이크는 request_generator 컨텍스트가 책임지고 닫음

# =======================
# 메인 실행
# =======================
def main():
    print("🎙️ 준비 완료. 웨이크워드를 말하세요 (예: '하이봇')")

    while True:
        try:
            # 1) 웨이크워드 대기
            for response in start_stream(is_command_mode=False):
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = normalize(result.alternatives[0].transcript.strip())
                if transcript:
                    print(" 인식:", transcript)

                # 웨이크워드 검사
                if any(normalize(w) in transcript for w in WAKE_WORDS):
                    # interim 중에도 어느 정도 안정화되면 넘어감
                    if result.is_final or result.stability >= 0.8:
                        print(" ✅ 웨이크워드 감지 → 명령을 말씀하세요 (3초)")
                        break  # 웨이크 스트림 탈출 → 명령 스트림으로 전환
            else:
                # 스트림이 특별한 이유로 종료되면 루프 재시작
                continue

            # 2) 명령어 인식 (한 문장)
            start_time = time.time()
            for cmd_response in start_stream(is_command_mode=True):
                if time.time() - start_time > 3.0:
                    print(" ⏰ 명령 대기 시간 초과")
                    break
                if not cmd_response.results:
                    continue
                cmd_result = cmd_response.results[0]
                if not cmd_result.alternatives:
                    continue

                cmd_text = normalize(cmd_result.alternatives[0].transcript.strip())
                if cmd_text:
                    print(" 명령 인식:", cmd_text)

                matched = False
                for cmd, variations in CMD_MAP.items():
                    if any(normalize(v) in cmd_text for v in variations):
                        matched = True
                        print(f" 👉 명령어: {cmd}")
                        # TODO: 여기서 실제 로봇 명령 호출
                        if cmd == "종료":
                            print(" 📴 시스템 종료")
                            return
                        break
                if not matched:
                    print(" ❓ 명령어를 인식하지 못했습니다.")
                break

        except KeyboardInterrupt:
            print("\n🛑 사용자 중단")
            break
        except Exception as e:
            # gRPC/네트워크 오류 등: 짧게 쉬고 재시작
            print(f"⚠️ 예외 발생, 재시작합니다: {e}")
            time.sleep(0.5)

if __name__ == "__main__":
    main()