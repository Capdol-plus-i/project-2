#!/usr/bin/env python3
"""
개선된 음성 인식 시스템
- 더 민감한 웨이크워드 감지
- 시각적 피드백 (컬러 출력)
- 설정 가능한 옵션들
- 로봇팔 연동 준비
- 에러 복구 로직
"""

import os
import sys
import time
import re
import unicodedata
import audioop
import json
import threading
import queue
from dataclasses import dataclass

from typing import Optional, Dict, List

import pyaudio
import webrtcvad
from google.cloud import speech

# ---- gRPC 경고 억제 ----
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_LOG_SEVERITY_LEVEL", "ERROR")

# ---- 컬러 출력 ----
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

@dataclass
class VoiceConfig:
    """음성 인식 설정"""
    # 오디오 설정
    rate: int = 16000
    frame_ms: int = 20
    vad_aggressiveness: int = 2

    # 웨이크워드 감지
    wake_stability_threshold: float = 0.5  # 0.8에서 0.5로 완화
    wake_confidence_threshold: float = 0.7

    # 명령 처리
    command_timeout: float = 5.0  # 3초에서 5초로 연장
    max_retries: int = 3

    # 마이크 설정
    mic_hint: str = "Blue Tiki"

    # 웨이크워드 & 명령어
    wake_words: Optional[List[str]] = None
    command_map: Optional[Dict[str, List[str]]] = None

    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["하이봇", "하이못", "아이봇", "AI봇", "아이", "하이"]

        if self.command_map is None:
            self.command_map = {
                # 조명 제어
                "조명켜": ["조명켜", "조명 켜", "라이트 온", "불켜", "켜줘"],
                "조명꺼": ["조명꺼", "조명 꺼", "라이트 오프", "불꺼", "꺼줘"],
                "밝게": ["밝게", "더 밝게", "밝기 올려", "업"],
                "어둡게": ["어둡게", "더 어둡게", "밝기 내려", "다운"],

                # 색상 변경
                "빨간색": ["빨간색", "빨강", "레드", "red"],
                "파란색": ["파란색", "파랑", "블루", "blue"],
                "녹색": ["녹색", "초록", "그린", "green"],
                "노란색": ["노란색", "노랑", "옐로우", "yellow"],
                "하얀색": ["하얀색", "흰색", "화이트", "white"],
                "무지개": ["무지개", "레인보우", "rainbow", "컬러풀"],

                # 팔로우 모드
                "팔로우시작": ["팔로우시작", "팔로우 시작", "따라와", "추적시작"],
                "팔로우정지": ["팔로우정지", "팔로우 정지", "멈춰", "추적정지"],
                "초기화": ["초기화", "리셋", "센터", "중앙", "홈"],

                # 시스템
                "종료": ["종료", "끝내", "종료해", "그만", "바이"]
            }

# 전역 설정
config = VoiceConfig()

# 계산된 상수들
SAMPLES_PER_FRAME = int(config.rate * config.frame_ms / 1000)
BYTES_PER_SAMPLE = 2
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", "", text)

def pick_input_device_index(p: pyaudio.PyAudio, hint: str | None = None) -> int | None:
    hint = (hint or "").lower()
    chosen = None
    print_colored("🎤 사용 가능한 마이크 장치:", Colors.CYAN)

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        name = f'{info.get("name","")}'
        host_api_index = int(info.get("hostApi", 0))
        host = f'{p.get_host_api_info_by_index(host_api_index).get("name","")}'

        is_match = hint and (hint in name.lower() or hint in host.lower())
        marker = "✓" if is_match else " "
        print(f"  {marker} [{i}] {name} ({host})")

        if is_match:
            return i
        if chosen is None:
            chosen = i

    return chosen

class ImprovedMicrophoneStream:
    """개선된 마이크 스트림 - 더 안정적인 에러 처리"""

    def __init__(self, target_rate, frame_ms, device_index=None):
        self._target_rate = target_rate
        self._frame_ms = frame_ms
        self._target_bytes_per_frame = int(target_rate * frame_ms / 1000) * BYTES_PER_SAMPLE

        self._buff = queue.Queue()
        self._carry = b""
        self._ratecv_state = None
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)

        self._device_index = device_index
        self._hw_rate = None
        self.closed = True
        self._error_count = 0

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        if self._device_index is None:
            self._device_index = pick_input_device_index(self._pa, config.mic_hint)
        if self._device_index is None:
            raise RuntimeError("입력 장치를 찾을 수 없습니다.")

        # 장치 정보 출력
        dinfo = self._pa.get_device_info_by_index(self._device_index)
        default_rate = dinfo.get("defaultSampleRate", 16000)
        hw_rate = int(round(float(default_rate)))

        # 다양한 샘플레이트로 시도
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
        print_colored(f"✓ 마이크 연결: {dinfo.get('name')} @ {self._hw_rate} Hz", Colors.GREEN)
        return self

    def __exit__(self, exc_type, value, traceback):
        try:
            if hasattr(self, '_stream'):
                self._stream.stop_stream()
                self._stream.close()
        except:
            pass
        finally:
            self.closed = True
            self._buff.put(None)
            if hasattr(self, '_pa'):
                self._pa.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        if status_flags:
            self._error_count += 1
            if self._error_count > 10:
                print_colored("⚠️ 마이크 오류가 너무 많습니다", Colors.WARNING)
        else:
            self._error_count = 0

        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def _to_target_rate(self, data: bytes) -> bytes:
        if self._hw_rate == self._target_rate:
            return data
        try:
            if self._hw_rate is not None:
                converted, self._ratecv_state = audioop.ratecv(
                    data, BYTES_PER_SAMPLE, 1, self._hw_rate, self._target_rate, self._ratecv_state
                )
            else:
                converted = data
            return converted
        except Exception:
            return data  # 변환 실패 시 원본 반환

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

                    try:
                        if self.vad.is_speech(frame, config.rate):
                            yield frame
                    except Exception:
                        # VAD 에러 시 프레임 건너뛰기
                        continue

        except GeneratorExit:
            return

class VoiceRecognitionEngine:
    """개선된 음성 인식 엔진"""

    def __init__(self):
        self.is_listening = False
        self.command_callback = None

    def set_command_callback(self, callback):
        """명령 실행 콜백 설정"""
        self.command_callback = callback

    def build_client_and_config(self, is_command_mode=False):
        client = speech.SpeechClient()

        phrases = list(config.wake_words)
        for v in config.command_map.values():
            phrases.extend(v)

        speech_context = speech.SpeechContext(phrases=phrases, boost=15.0)

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=config.rate,
            language_code="ko-KR",
            speech_contexts=[speech_context],
            model="command_and_search",
            enable_automatic_punctuation=True,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=not is_command_mode,
            single_utterance=is_command_mode
        )
        return client, streaming_config

    def start_stream(self, is_command_mode=False, device_index=None):
        client, streaming_config = self.build_client_and_config(is_command_mode=is_command_mode)
        mic = ImprovedMicrophoneStream(config.rate, config.frame_ms, device_index=device_index)
        mic.__enter__()
        audio_generator = mic.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=frame) for frame in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        return mic, responses

    def execute_command(self, command: str):
        """명령 실행"""
        print_colored(f"🤖 명령 실행: {command}", Colors.BOLD)

        if self.command_callback:
            try:
                self.command_callback(command)
            except Exception as e:
                print_colored(f"❌ 명령 실행 실패: {e}", Colors.FAIL)
        else:
            # 기본 동작
            if command == "종료":
                print_colored("시스템을 종료합니다.", Colors.WARNING)
                return False
            else:
                print_colored(f"'{command}' 명령을 처리했습니다.", Colors.GREEN)

        return True

    def listen_for_wake_word(self, retries=0):
        """웨이크워드 대기"""
        try:
            if retries == 0:
                print_colored("🎙️ 웨이크워드를 말하세요...", Colors.CYAN)
            else:
                print_colored(f"🔄 재시도 중... ({retries}/{config.max_retries})", Colors.WARNING)

            stream, responses = self.start_stream(is_command_mode=False)

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = normalize(result.alternatives[0].transcript.strip())
                confidence = result.alternatives[0].confidence if hasattr(result.alternatives[0], 'confidence') else 0.0

                if transcript:
                    # 실시간 인식 표시
                    if result.is_final:
                        print_colored(f"  📝 최종: {transcript} (신뢰도: {confidence:.2f})", Colors.BLUE)
                    else:
                        print(f"\r  📝 인식중: {transcript}", end="", flush=True)

                # 웨이크워드 검사 - 더 관대한 조건
                wake_detected = any(normalize(w) in transcript for w in config.wake_words)

                if wake_detected:
                    is_confident = (
                        (result.is_final and confidence >= config.wake_confidence_threshold) or
                        (result.stability and result.stability >= config.wake_stability_threshold) or
                        (confidence >= 0.8)  # 높은 신뢰도면 즉시 반응
                    )

                    if is_confident:
                        print_colored("\n✅ 웨이크워드 감지! 명령을 말씀하세요...", Colors.GREEN)
                        stream.__exit__(None, None, None)
                        return True

            stream.__exit__(None, None, None)
            return False

        except Exception as e:
            print_colored(f"❌ 웨이크워드 감지 오류: {e}", Colors.FAIL)
            if retries < config.max_retries:
                time.sleep(1)
                return self.listen_for_wake_word(retries + 1)
            return False

    def listen_for_command(self, retries=0):
        """명령어 대기"""
        try:
            cmd_stream, cmd_responses = self.start_stream(is_command_mode=True)
            start_time = time.time()

            print_colored(f"⏱️ 명령 대기 중... ({config.command_timeout}초)", Colors.CYAN)

            for cmd_response in cmd_responses:
                elapsed = time.time() - start_time
                if elapsed > config.command_timeout:
                    print_colored("\n⏰ 명령 대기 시간 초과", Colors.WARNING)
                    break

                if not cmd_response.results:
                    continue

                cmd_result = cmd_response.results[0]
                if not cmd_result.alternatives:
                    continue

                cmd_transcript = normalize(cmd_result.alternatives[0].transcript.strip())

                if cmd_transcript:
                    print_colored(f"  📝 명령: {cmd_transcript}", Colors.BLUE)

                    # 명령어 매칭
                    for cmd, variations in config.command_map.items():
                        if any(normalize(v) in cmd_transcript for v in variations):
                            cmd_stream.__exit__(None, None, None)
                            return cmd

            cmd_stream.__exit__(None, None, None)

            if retries < config.max_retries:
                print_colored(f"❓ 명령을 인식하지 못했습니다. 다시 시도합니다. ({retries+1}/{config.max_retries})", Colors.WARNING)
                return self.listen_for_command(retries + 1)
            else:
                print_colored("❌ 명령 인식에 실패했습니다.", Colors.FAIL)
                return None

        except Exception as e:
            print_colored(f"❌ 명령 인식 오류: {e}", Colors.FAIL)
            if retries < config.max_retries:
                time.sleep(1)
                return self.listen_for_command(retries + 1)
            return None

    def run(self):
        """메인 루프"""
        print_colored("🎤 개선된 음성 인식 시스템 시작", Colors.HEADER + Colors.BOLD)
        print_colored("=" * 50, Colors.HEADER)

        # 설정 정보 출력
        print_colored(f"웨이크워드: {', '.join(config.wake_words)}", Colors.CYAN)
        print_colored(f"명령어: {', '.join(config.command_map.keys())}", Colors.CYAN)
        print_colored(f"감지 임계값: {config.wake_stability_threshold}", Colors.CYAN)
        print_colored("=" * 50, Colors.HEADER)

        self.is_listening = True

        try:
            while self.is_listening:
                # 1. 웨이크워드 대기
                if self.listen_for_wake_word():
                    time.sleep(0.1)  # 짧은 딜레이

                    # 2. 명령어 대기
                    command = self.listen_for_command()

                    if command:
                        # 3. 명령 실행
                        if not self.execute_command(command):
                            break  # 종료 명령

                    print_colored("\n" + "="*30, Colors.HEADER)
                    time.sleep(0.5)  # 다음 사이클 전 딜레이

        except KeyboardInterrupt:
            print_colored("\n^C 사용자 중단", Colors.WARNING)
        except Exception as e:
            print_colored(f"시스템 오류: {e}", Colors.FAIL)
        finally:
            self.is_listening = False
            print_colored("🛑 음성 인식 시스템 종료", Colors.GREEN)

# NeoPixel + 로봇팔 제어 콜백
def robot_command_callback(command: str):
    """NeoPixel 조명 + 로봇팔 팔로우 명령 실행"""

    # 조명 제어 명령 (짧은 명령어)
    light_commands = {
        "조명켜": lambda: send_arduino_command("ON"),
        "조명꺼": lambda: send_arduino_command("OFF"),
        "밝게": lambda: send_arduino_command("UP"),
        "어둡게": lambda: send_arduino_command("DOWN"),

        "빨간색": lambda: send_arduino_command("R"),
        "파란색": lambda: send_arduino_command("B"),
        "녹색": lambda: send_arduino_command("G"),
        "노란색": lambda: send_arduino_command("Y"),
        "하얀색": lambda: send_arduino_command("W"),
        "무지개": lambda: send_arduino_command("RAINBOW"),
    }

    # 팔로우 제어 명령
    follow_commands = {
        "팔로우시작": lambda: start_hand_following(),
        "팔로우정지": lambda: stop_hand_following(),
        "초기화": lambda: reset_arm_position(),
    }

    if command in light_commands:
        print_colored(f"💡 조명 제어: {command}", Colors.CYAN)
        light_commands[command]()
    elif command in follow_commands:
        print_colored(f"🦾 로봇팔 제어: {command}", Colors.GREEN)
        follow_commands[command]()
    else:
        print_colored(f"❓ 알 수 없는 명령: {command}", Colors.WARNING)

def send_arduino_command(cmd: str):
    """Arduino에 NeoPixel 제어 명령 전송"""
    try:
        import serial
        import time

        print_colored(f"  📡 Arduino 명령 전송: {cmd}", Colors.BLUE)

        # Arduino 시리얼 연결
        with serial.Serial('/dev/arduino', 9600, timeout=2) as ser:
            time.sleep(0.1)  # Arduino 준비 시간

            # 명령 전송
            command_line = f"{cmd}\n"
            ser.write(command_line.encode('utf-8'))
            ser.flush()

            # 응답 대기 (안전한 디코딩)
            time.sleep(0.2)
            while ser.in_waiting > 0:
                try:
                    raw_data = ser.readline()
                    # UTF-8 디코딩 시도
                    try:
                        response = raw_data.decode('utf-8', errors='ignore').strip()
                    except UnicodeDecodeError:
                        # UTF-8 실패 시 Latin-1로 시도
                        response = raw_data.decode('latin-1', errors='ignore').strip()

                    if response and len(response) > 0:
                        # 비 ASCII 문자 제거
                        clean_response = ''.join(char for char in response if ord(char) < 128)
                        if clean_response:
                            print_colored(f"    → Arduino 응답: {clean_response}", Colors.GREEN)
                except Exception:
                    pass  # 응답 수신 실패는 무시

            print_colored(f"  ✓ 명령 전송 완료", Colors.GREEN)

    except serial.SerialException as e:
        print_colored(f"  ❌ Arduino 연결 실패: {e}", Colors.FAIL)
        print_colored(f"    확인사항: /dev/arduino 장치가 연결되어 있는지 확인하세요", Colors.WARNING)
    except ImportError:
        print_colored(f"  ❌ pyserial 모듈이 설치되지 않음: pip install pyserial", Colors.FAIL)
    except Exception as e:
        print_colored(f"  ❌ Arduino 통신 실패: {e}", Colors.FAIL)

def start_hand_following():
    """손 추적 모드 시작"""
    try:
        # TODO: MediaPipe 손 좌표를 구독하고 로봇팔 동기화 시작
        print_colored("  👋 손 추적 모드 시작 - 작업자의 손을 따라 조명이 움직입니다", Colors.GREEN)
        # 예시:
        # hand_tracker.start_following()
        # robot_arm.enable_follow_mode()
    except Exception as e:
        print_colored(f"  ❌ 손 추적 시작 실패: {e}", Colors.FAIL)

def stop_hand_following():
    """손 추적 모드 정지"""
    try:
        print_colored("  ✋ 손 추적 모드 정지", Colors.WARNING)
        # TODO: 손 추적 정지
        # hand_tracker.stop_following()
        # robot_arm.disable_follow_mode()
    except Exception as e:
        print_colored(f"  ❌ 손 추적 정지 실패: {e}", Colors.FAIL)

def reset_arm_position():
    """로봇팔 초기 위치로 복귀"""
    try:
        print_colored("  🏠 로봇팔 초기 위치로 복귀", Colors.CYAN)
        # TODO: 로봇팔을 중앙/초기 위치로 이동
        # robot_arm.move_to_home_position()
    except Exception as e:
        print_colored(f"  ❌ 초기화 실패: {e}", Colors.FAIL)

def main():
    engine = VoiceRecognitionEngine()
    engine.set_command_callback(robot_command_callback)
    engine.run()

if __name__ == "__main__":
    main()