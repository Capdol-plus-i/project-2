#!/usr/bin/env python3
"""
음성 제어 Arduino 컨트롤러
voice_recognition.py와 arduino_terminal.py를 통합하여 음성 명령으로 Arduino 제어
"""

import os
import sys
import time
import re
import unicodedata
import audioop
import serial
import threading
from six.moves import queue

import pyaudio
import webrtcvad
from google.cloud import speech

# ---- gRPC 경고 억제 ----
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_LOG_SEVERITY_LEVEL", "ERROR")

# =======================
# 음성 인식 설정
# =======================
RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)
BYTES_PER_SAMPLE = 2
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE

# 웨이크워드와 Arduino 명령어 매핑
WAKE_WORDS = ["아두이노", "아두", "아까", "아드"]

ARDUINO_CMD_MAP = {
    # 조명 제어
    "켜": ["켜", "켜줘", "온", "라이트온", "불켜", "불켜줘"],
    "꺼": ["꺼", "꺼줘", "오프", "라이트오프", "불꺼", "불꺼줘"],
    "밝게": ["밝게", "밝게해", "밝게해줘", "업", "올려", "올려줘"],
    "어둡게": ["어둡게", "어둡게해", "어둡게해줘", "다운", "내려", "내려줘"],

    # 색상 제어
    "빨간색": ["빨간색", "빨강", "빨갛게", "레드", "빨개"],
    "파란색": ["파란색", "파랑", "파랗게", "블루", "파래"],
    "녹색": ["녹색", "초록", "초록색", "그린", "초록해"],
    "노란색": ["노란색", "노랑", "노랗게", "옐로우", "노래"],
    "흰색": ["흰색", "하얀색", "하얗게", "화이트", "하얘"],
    "무지개": ["무지개", "레인보우", "무지갯빛", "알록달록"],

    # 시스템 제어
    "상태": ["상태", "상태확인", "스테이터스", "어떤상태"],
    "리셋": ["리셋", "초기화", "재시작"],
    "하트비트": ["하트비트", "연결확인", "핑"],

    # 종료
    "종료": ["종료", "끝내", "종료해", "나가", "나가줘", "그만"]
}

# 장치 선택 힌트
MIC_HINT = os.environ.get("MIC_HINT", "Blue Tiki")

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")

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
    def __init__(self, target_rate, frame_ms, device_index=None):
        self._target_rate = target_rate
        self._frame_ms = frame_ms
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
            raise RuntimeError("입력 장치를 찾을 수 없습니다.")

        dinfo = self._pa.get_device_info_by_index(self._device_index)
        hw_rate = int(round(dinfo.get("defaultSampleRate", 16000)))
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
        print_colored(f"🎤 마이크 연결됨 @ {self._hw_rate} Hz", Colors.GREEN)
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
        if self._hw_rate == self._target_rate:
            return data
        converted, self._ratecv_state = audioop.ratecv(
            data, BYTES_PER_SAMPLE, 1, self._hw_rate, self._target_rate, self._ratecv_state
        )
        return converted

    def generator(self):
        try:
            while not self.closed:
                chunk = self._buff.get()
                if chunk is None:
                    return
                pcm16k = self._to_target_rate(chunk)
                self._carry += pcm16k
                while len(self._carry) >= self._target_bytes_per_frame:
                    frame = self._carry[:self._target_bytes_per_frame]
                    self._carry = self._carry[self._target_bytes_per_frame:]
                    if self.vad.is_speech(frame, RATE):
                        yield frame
        except GeneratorExit:
            return

class ArduinoController:
    def __init__(self):
        self.ser = None
        self.listening = False
        self.listener_thread = None

    def connect(self):
        try:
            print_colored("🔌 Arduino 연결 중...", Colors.CYAN)
            self.ser = serial.Serial('/dev/arduino', 9600, timeout=1)
            time.sleep(2)
            print_colored("✓ Arduino 연결 성공!", Colors.GREEN)
            self.send_command("STATUS")
            return True
        except serial.SerialException as e:
            print_colored(f"❌ Arduino 연결 실패: {e}", Colors.FAIL)
            return False
        except Exception as e:
            print_colored(f"❌ 예상치 못한 오류: {e}", Colors.FAIL)
            return False

    def start_listening(self):
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen_responses, daemon=True)
        self.listener_thread.start()

    def _listen_responses(self):
        while self.listening and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    raw_data = self.ser.readline()
                    try:
                        response = raw_data.decode('utf-8', errors='ignore').strip()
                    except UnicodeDecodeError:
                        response = raw_data.decode('latin-1', errors='ignore').strip()

                    if response and len(response) > 0:
                        clean_response = ''.join(char for char in response if ord(char) < 128)
                        if clean_response:
                            print_colored(f"📨 Arduino: {clean_response}", Colors.GREEN)
                time.sleep(0.1)
            except Exception as e:
                if self.listening:
                    print_colored(f"❌ 수신 오류: {e}", Colors.FAIL)
                continue

    def send_command(self, command):
        if not self.ser or not self.ser.is_open:
            print_colored("❌ Arduino가 연결되지 않음", Colors.FAIL)
            return False

        try:
            print_colored(f"📡 Arduino 명령 실행: {command}", Colors.BLUE)
            self.ser.write(f"{command}\n".encode('utf-8'))
            self.ser.flush()
            return True
        except Exception as e:
            print_colored(f"❌ 전송 실패: {e}", Colors.FAIL)
            return False

    def disconnect(self):
        self.listening = False
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print_colored("🔌 Arduino 연결 종료", Colors.GREEN)

class VoiceArduinoController:
    def __init__(self):
        self.arduino = ArduinoController()
        self.running = False

    def map_voice_to_arduino_command(self, voice_text):
        """음성 명령을 Arduino 명령으로 변환"""
        voice_normalized = normalize(voice_text)

        command_mapping = {
            "켜": "ON",
            "꺼": "OFF",
            "밝게": "UP",
            "어둡게": "DOWN",
            "빨간색": "R",
            "파란색": "B",
            "녹색": "G",
            "노란색": "Y",
            "흰색": "W",
            "무지개": "RAINBOW",
            "상태": "STATUS",
            "리셋": "RESET",
            "하트비트": "HEARTBEAT"
        }

        for cmd, arduino_cmd in command_mapping.items():
            for variation in ARDUINO_CMD_MAP[cmd]:
                if normalize(variation) in voice_normalized:
                    return arduino_cmd

        return None

    def build_client_and_config(self, is_command_mode=False):
        client = speech.SpeechClient()

        phrases = list(WAKE_WORDS)
        for variations in ARDUINO_CMD_MAP.values():
            phrases.extend(variations)

        speech_context = speech.SpeechContext(phrases=phrases, boost=15.0)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
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

    def start_stream(self, is_command_mode=False, device_index=None):
        client, streaming_config = self.build_client_and_config(is_command_mode=is_command_mode)
        mic = MicrophoneStream(RATE, FRAME_MS, device_index=device_index)
        mic.__enter__()
        audio_generator = mic.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=frame) for frame in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        return mic, responses

    def start_voice_control(self):
        print_colored("🎙️ 음성 제어 시작. 웨이크워드를 말하세요 (예: '아두이노')", Colors.CYAN)
        self.running = True

        try:
            while self.running:
                stream, responses = self.start_stream(is_command_mode=False)
                for response in responses:
                    if not response.results:
                        continue
                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    transcript = normalize(result.alternatives[0].transcript.strip())
                    if transcript:
                        print_colored(f" 🎯 인식: {result.alternatives[0].transcript}", Colors.WARNING)

                    if any(normalize(w) in transcript for w in WAKE_WORDS):
                        if result.is_final or result.stability > 0.8:
                            print_colored(" ✅ 웨이크워드 감지 → Arduino 명령을 말씀하세요.", Colors.GREEN)
                            stream.__exit__(None, None, None)
                            time.sleep(0.05)

                            cmd_stream, cmd_responses = self.start_stream(is_command_mode=True)
                            start_time = time.time()
                            MAX_COMMAND_DURATION = 3.0

                            for cmd_response in cmd_responses:
                                if time.time() - start_time > MAX_COMMAND_DURATION:
                                    print_colored(" ⏰ 명령 대기 초과", Colors.WARNING)
                                    cmd_stream.__exit__(None, None, None)
                                    break

                                if not cmd_response.results:
                                    continue
                                cmd_result = cmd_response.results[0]
                                if not cmd_result.alternatives:
                                    continue
                                cmd_transcript = cmd_result.alternatives[0].transcript.strip()
                                if cmd_transcript:
                                    print_colored(f" 🎯 명령: {cmd_transcript}", Colors.CYAN)

                                # Arduino 명령으로 변환
                                arduino_cmd = self.map_voice_to_arduino_command(cmd_transcript)

                                if arduino_cmd:
                                    if arduino_cmd == "OFF" and "종료" in normalize(cmd_transcript):
                                        print_colored(" 👋 시스템 종료", Colors.WARNING)
                                        self.running = False
                                        cmd_stream.__exit__(None, None, None)
                                        break
                                    else:
                                        print_colored(f" 👉 Arduino 명령어: {arduino_cmd}", Colors.GREEN)
                                        self.arduino.send_command(arduino_cmd)
                                        cmd_stream.__exit__(None, None, None)
                                        break
                                else:
                                    # 종료 명령 확인
                                    if any(normalize(variation) in normalize(cmd_transcript)
                                           for variation in ARDUINO_CMD_MAP["종료"]):
                                        print_colored(" 👋 시스템 종료", Colors.WARNING)
                                        self.running = False
                                        cmd_stream.__exit__(None, None, None)
                                        break
                                    else:
                                        print_colored(" ❓ 알 수 없는 명령. 다시 말씀해주세요.", Colors.FAIL)
                                break
                            break
        except KeyboardInterrupt:
            print_colored("\n^C 종료", Colors.WARNING)
        except Exception as e:
            print_colored(f"에러: {e}", Colors.FAIL)

def main():
    print_colored("🤖 음성 제어 Arduino 컨트롤러", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)

    controller = VoiceArduinoController()

    # Arduino 연결
    if not controller.arduino.connect():
        print_colored("Arduino 연결에 실패했습니다. 프로그램을 종료합니다.", Colors.FAIL)
        sys.exit(1)

    # Arduino 응답 수신 시작
    controller.arduino.start_listening()

    print_colored("\n📋 지원되는 음성 명령:", Colors.CYAN)
    print_colored("• 조명: '켜', '꺼', '밝게', '어둡게'", Colors.ENDC)
    print_colored("• 색상: '빨간색', '파란색', '녹색', '노란색', '흰색', '무지개'", Colors.ENDC)
    print_colored("• 시스템: '상태', '리셋', '하트비트'", Colors.ENDC)
    print_colored("• 종료: '종료', '끝내'", Colors.ENDC)
    print_colored("\n사용법: '아두이노' 웨이크워드 후 명령어 말하기", Colors.WARNING)
    print_colored("예시: '아두이노' → '빨간색'", Colors.WARNING)
    print_colored("종료: Ctrl+C 또는 '아두이노' → '종료'", Colors.WARNING)

    try:
        # 음성 제어 시작
        controller.start_voice_control()
    except KeyboardInterrupt:
        print_colored("\n\n🛑 프로그램을 종료합니다.", Colors.WARNING)
    except Exception as e:
        print_colored(f"\n❌ 예상치 못한 오류: {e}", Colors.FAIL)
    finally:
        controller.arduino.disconnect()

if __name__ == "__main__":
    main()