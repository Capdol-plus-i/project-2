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
import signal
import weakref
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, Dict, List

import pyaudio
import webrtcvad
from google.cloud import speech

# Arduino Terminal 통합
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ pyserial 모듈이 없습니다. Arduino 제어가 비활성화됩니다.")

# Arduino Terminal 클래스 (arduino_terminal.py에서 가져옴)
class ArduinoTerminal:
    def __init__(self):
        self.ser = None
        self.listening = False
        self.listener_thread = None

    def connect(self):
        """Arduino 연결"""
        if not SERIAL_AVAILABLE:
            print_colored("❌ pyserial 모듈이 필요합니다: pip install pyserial", Colors.FAIL)
            return False

        try:
            print_colored("🔌 Arduino 연결 중...", Colors.CYAN)
            self.ser = serial.Serial('/dev/arduino', 9600, timeout=1)
            time.sleep(2)  # Arduino 부팅 대기
            print_colored("✓ Arduino 연결 성공!", Colors.GREEN)

            # 연결 후 상태 확인
            self.send_command("STATUS")
            return True

        except serial.SerialException as e:
            print_colored(f"❌ Arduino 연결 실패: {e}", Colors.FAIL)
            print_colored("확인사항: /dev/arduino 장치가 연결되어 있는지 확인하세요", Colors.WARNING)
            return False
        except Exception as e:
            print_colored(f"❌ 예상치 못한 오류: {e}", Colors.FAIL)
            return False

    def send_command(self, command):
        """Arduino에 명령 전송"""
        if not self.ser or not self.ser.is_open:
            print_colored("❌ Arduino가 연결되지 않음", Colors.FAIL)
            return False

        try:
            print_colored(f"📡 전송: {command}", Colors.BLUE)
            self.ser.write(f"{command}\n".encode('utf-8'))
            self.ser.flush()

            # 응답 대기
            time.sleep(0.2)
            if self.ser.in_waiting > 0:
                try:
                    raw_data = self.ser.readline()
                    response = raw_data.decode('utf-8', errors='ignore').strip()
                    if response:
                        clean_response = ''.join(char for char in response if ord(char) < 128)
                        if clean_response:
                            print_colored(f"📨 Arduino: {clean_response}", Colors.GREEN)
                except Exception:
                    pass

            return True

        except Exception as e:
            print_colored(f"❌ 전송 실패: {e}", Colors.FAIL)
            return False

    def disconnect(self):
        """Arduino 연결 종료"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print_colored("🔌 Arduino 연결 종료", Colors.GREEN)

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

    # 세션 모드 (웨이크워드 후 연속 명령)
    session_mode: bool = True
    session_timeout: float = 30.0  # 30초 동안 연속 명령 가능
    session_idle_timeout: float = 10.0  # 10초 무음 시 세션 종료

    # 연속 모드 (웨이크워드 없이 항상 명령어 대기)
    continuous_mode: bool = True  # True: 연속 모드, False: 웨이크워드 모드
    continuous_timeout: float = 2.0  # 연속 모드에서 명령어 대기 시간

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
                "종료": ["종료", "끝내", "종료해", "그만", "바이"],
                "아두이노연결": ["아두이노연결", "아두이노 연결", "Arduino 연결", "연결확인"],
                "상태확인": ["상태확인", "상태 확인", "STATUS", "status"],
                "웨이크워드모드": ["웨이크워드모드", "웨이크워드 모드", "웨이크모드"],
                "연속모드": ["연속모드", "연속 모드", "계속모드"]
            }

# 전역 설정
config = VoiceConfig()

# 계산된 상수들 (성능 최적화)
SAMPLES_PER_FRAME = int(config.rate * config.frame_ms / 1000)
BYTES_PER_SAMPLE = 2
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE

# 전역 캐시 변수들
_normalized_wake_words = None
_normalized_commands = None

def get_normalized_wake_words():
    """normalize된 웨이크워드 캐시"""
    global _normalized_wake_words
    if _normalized_wake_words is None:
        _normalized_wake_words = [normalize(w) for w in config.wake_words]
    return _normalized_wake_words

def get_normalized_commands():
    """normalize된 명령어 맵 캐시"""
    global _normalized_commands
    if _normalized_commands is None:
        _normalized_commands = {}
        for cmd, variations in config.command_map.items():
            _normalized_commands[cmd] = [normalize(v) for v in variations]
    return _normalized_commands

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

        self._buff = queue.Queue(maxsize=100)  # 큐 크기 제한으로 메모리 사용량 제어
        self._carry = b""
        self._ratecv_state = None
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)

        self._device_index = device_index
        self._hw_rate = None
        self.closed = True
        self._error_count = 0
        self._cleanup_lock = threading.Lock()  # 정리 작업 동기화
        self._pa = None
        self._stream = None

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
        with self._cleanup_lock:
            if self.closed:
                return

            self.closed = True

            # 스트림 정리
            if self._stream:
                try:
                    if not self._stream.is_stopped():
                        self._stream.stop_stream()
                except:
                    pass
                try:
                    self._stream.close()
                except:
                    pass
                self._stream = None

            # PyAudio 정리
            if self._pa:
                try:
                    self._pa.terminate()
                except:
                    pass
                self._pa = None

            # 큐 정리 - 논블로킹으로 비우기
            try:
                while not self._buff.empty():
                    try:
                        self._buff.get_nowait()
                    except queue.Empty:
                        break
                self._buff.put(None)  # 종료 신호
            except:
                pass

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        if self.closed:
            return (None, pyaudio.paComplete)

        if status_flags:
            self._error_count += 1
            if self._error_count > 10:
                print_colored("⚠️ 마이크 오류가 너무 많습니다", Colors.WARNING)
                return (None, pyaudio.paComplete)
        else:
            self._error_count = 0

        try:
            # 큐가 가득 찬 경우 오래된 데이터 제거
            if self._buff.full():
                try:
                    self._buff.get_nowait()
                except queue.Empty:
                    pass
            self._buff.put(in_data, block=False)
        except queue.Full:
            pass  # 큐가 가득 차면 프레임 건너뛰기

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
                try:
                    chunk = self._buff.get(timeout=1.0)  # 타임아웃 추가
                    if chunk is None or self.closed:
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
                            continue

                except queue.Empty:
                    continue  # 타임아웃 시 계속 시도
                except Exception:
                    break

        except GeneratorExit:
            return
        finally:
            # 정리 작업
            self._carry = b""

class VoiceRecognitionEngine:
    """개선된 음성 인식 엔진"""

    def __init__(self):
        self.is_listening = False
        self.command_callback = None
        self.arduino_terminal = ArduinoTerminal()
        self.arduino_connected = False
        self._active_streams = weakref.WeakSet()
        self._shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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

    def _signal_handler(self, signum, frame):
        """시그널 핸들러 - 깨끗한 종료"""
        print_colored("\n🛝 종료 신호 수신, 시스템 종료 중...", Colors.WARNING)
        self._shutdown_event.set()
        self.is_listening = False

    @contextmanager
    def _managed_stream(self, is_command_mode=False, device_index=None):
        """스트림 자원 관리"""
        mic = None
        try:
            client, streaming_config = self.build_client_and_config(is_command_mode=is_command_mode)
            mic = ImprovedMicrophoneStream(config.rate, config.frame_ms, device_index=device_index)
            mic.__enter__()
            self._active_streams.add(mic)

            audio_generator = mic.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=frame) for frame in audio_generator)
            responses = client.streaming_recognize(streaming_config, requests)

            yield mic, responses

        finally:
            if mic:
                try:
                    mic.__exit__(None, None, None)
                except:
                    pass
                try:
                    self._active_streams.discard(mic)
                except:
                    pass

    def start_stream(self, is_command_mode=False, device_index=None):
        """기존 호환성을 위한 래퍼"""
        with self._managed_stream(is_command_mode, device_index) as (mic, responses):
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

    def listen_for_command(self, retries=0, timeout=None, continuous_mode=False):
        """명령어 대기"""
        try:
            cmd_stream, cmd_responses = self.start_stream(is_command_mode=not continuous_mode)
            start_time = time.time()

            # 타임아웃 설정 (기본값은 config의 command_timeout)
            command_timeout = timeout if timeout is not None else config.command_timeout

            if not continuous_mode:
                print_colored(f"⏱️ 명령 대기 중... ({command_timeout}초)", Colors.CYAN)

            last_transcript = ""

            for cmd_response in cmd_responses:
                elapsed = time.time() - start_time
                if elapsed > command_timeout:
                    if not continuous_mode:
                        print_colored("\n⏰ 명령 대기 시간 초과", Colors.WARNING)
                    break

                if not cmd_response.results:
                    continue

                cmd_result = cmd_response.results[0]
                if not cmd_result.alternatives:
                    continue

                cmd_transcript = normalize(cmd_result.alternatives[0].transcript.strip())

                if cmd_transcript and cmd_transcript != last_transcript:
                    if continuous_mode:
                        # 연속 모드에서는 실시간 표시
                        if cmd_result.is_final:
                            print_colored(f"  📝 인식: {cmd_transcript}", Colors.BLUE)
                        else:
                            print(f"\r  📝 듣는 중: {cmd_transcript}", end="", flush=True)
                    else:
                        print_colored(f"  📝 명령: {cmd_transcript}", Colors.BLUE)

                    # 명령어 매칭 (final 결과에서만)
                    if cmd_result.is_final or not continuous_mode:
                        for cmd, variations in config.command_map.items():
                            if any(normalize(v) in cmd_transcript for v in variations):
                                cmd_stream.__exit__(None, None, None)
                                if continuous_mode:
                                    print()  # 줄바꿈
                                return cmd

                        # 연속 모드에서 잘못된 명령은 무시하고 계속
                        if continuous_mode and cmd_result.is_final:
                            print()  # 줄바꿈
                            print_colored(f"  ❓ 알 수 없는 명령: {cmd_transcript}", Colors.WARNING)
                            # 스트림을 유지하고 계속 듣기
                            start_time = time.time()  # 타이머 리셋

                    last_transcript = cmd_transcript

            cmd_stream.__exit__(None, None, None)

            # 재시도는 상위 함수에서 처리
            return None

        except Exception as e:
            print_colored(f"❌ 명령 인식 오류: {e}", Colors.FAIL)
            # 재시도는 상위 함수에서 처리
            return None

    def run(self):
        """메인 루프"""
        print_colored("🎤 개선된 음성 인식 시스템 시작", Colors.HEADER + Colors.BOLD)
        print_colored("=" * 50, Colors.HEADER)

        # 설정 정보 출력
        mode_text = "연속 모드 (웨이크워드 불필요)" if config.continuous_mode else "웨이크워드 모드"
        print_colored(f"인식 모드: {mode_text}", Colors.CYAN + Colors.BOLD)

        if not config.continuous_mode:
            wake_words = config.wake_words or []
            print_colored(f"웨이크워드: {', '.join(wake_words)}", Colors.CYAN)
            print_colored(f"감지 임계값: {config.wake_stability_threshold}", Colors.CYAN)
            if config.session_mode:
                print_colored(f"세션 모드: 활성화 (세션 시간: {config.session_timeout}초, 무음 종료: {config.session_idle_timeout}초)", Colors.CYAN)
        else:
            print_colored(f"연속 인식 타임아웃: {config.continuous_timeout}초", Colors.CYAN)

        command_keys = list(config.command_map.keys()) if config.command_map else []
        print_colored(f"사용 가능한 명령어: {', '.join(command_keys[:8])}...", Colors.CYAN)
        print_colored("💡 '웨이크워드모드' 또는 '연속모드'로 모드 전환 가능", Colors.WARNING)
        print_colored("=" * 50, Colors.HEADER)

        self.is_listening = True

        try:
            while self.is_listening:
                if config.continuous_mode:
                    # 연속 모드: 웨이크워드 없이 항상 명령어 대기
                    self._handle_continuous_mode()
                else:
                    # 웨이크워드 모드: 기존 방식
                    if self.listen_for_wake_word():
                        time.sleep(0.1)  # 짧은 딜레이

                        if config.session_mode:
                            # 세션 모드: 연속 명령 처리
                            self._handle_session_mode()
                        else:
                            # 기존 모드: 올바른 명령어가 들어올 때까지 계속 시도
                            self._handle_single_command_mode()

                        print_colored("\n" + "="*30, Colors.HEADER)
                        time.sleep(0.5)  # 다음 사이클 전 딜레이

        except KeyboardInterrupt:
            print_colored("\n^C 사용자 중단", Colors.WARNING)
        except Exception as e:
            print_colored(f"시스템 오류: {e}", Colors.FAIL)
        finally:
            self._cleanup_resources()
            print_colored("🛑 음성 인식 시스템 종료", Colors.GREEN)

    def _cleanup_resources(self):
        """리소스 정리"""
        self.is_listening = False
        self._shutdown_event.set()

        # 모든 액티브 스트림 정리
        for stream in list(self._active_streams):
            try:
                stream.__exit__(None, None, None)
            except:
                pass

        # Arduino 연결 정리
        if self.arduino_terminal:
            try:
                self.arduino_terminal.disconnect()
            except:
                pass

    def _handle_session_mode(self):
        """세션 모드 처리 - 웨이크워드 후 연속 명령"""
        print_colored("🎯 세션 모드 시작 - 연속 명령을 받을 수 있습니다", Colors.GREEN + Colors.BOLD)
        print_colored("💡 '종료', '세션종료' 또는 무음으로 세션을 종료할 수 있습니다", Colors.CYAN)

        session_start = time.time()
        last_activity = time.time()

        while True:
            # 세션 타임아웃 체크
            if time.time() - session_start > config.session_timeout:
                print_colored(f"⏰ 세션 시간 초과 ({config.session_timeout}초)", Colors.WARNING)
                break

            # 무음 타임아웃 체크
            if time.time() - last_activity > config.session_idle_timeout:
                print_colored(f"😴 무음으로 인한 세션 종료 ({config.session_idle_timeout}초)", Colors.WARNING)
                break

            # 명령어 대기 (짧은 타임아웃으로)
            command = self.listen_for_command(timeout=3.0)

            if command:
                last_activity = time.time()

                # 세션 종료 명령 체크
                if command in ["종료", "세션종료", "exit", "quit"]:
                    print_colored("👋 세션을 종료합니다", Colors.GREEN)
                    break

                # 시스템 종료 명령
                if not self.execute_command(command):
                    self.is_listening = False
                    break

                print_colored("🎧 다음 명령을 기다리는 중...", Colors.BLUE)
            else:
                # 명령이 없으면 계속 시도 (세션 모드에서는 더 관대하게)
                print_colored("🔄 명령을 다시 말씀해주세요...", Colors.CYAN)
                time.sleep(0.5)

        print_colored("🏁 세션 모드 종료", Colors.GREEN)

    def _handle_single_command_mode(self):
        """단일 명령 모드 처리 - 올바른 명령어가 들어올 때까지 계속 시도"""
        print_colored("🎯 명령어를 말씀하세요", Colors.GREEN + Colors.BOLD)
        print_colored("💡 올바른 명령어가 인식될 때까지 계속 시도합니다", Colors.CYAN)

        max_attempts = 5  # 최대 시도 횟수
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print_colored(f"🎤 명령어 대기 중... ({attempt}/{max_attempts})", Colors.BLUE)

            command = self.listen_for_command(retries=0, timeout=config.command_timeout)

            if command:
                print_colored(f"✅ 명령어 인식됨: {command}", Colors.GREEN)
                if not self.execute_command(command):
                    self.is_listening = False  # 종료 명령
                break
            else:
                if attempt < max_attempts:
                    print_colored(f"❓ 명령어를 인식하지 못했습니다. 다시 시도해주세요. ({attempt}/{max_attempts})", Colors.WARNING)
                    time.sleep(0.5)  # 짧은 대기
                else:
                    print_colored("❌ 최대 시도 횟수를 초과했습니다. 웨이크워드부터 다시 시작합니다.", Colors.FAIL)

        print_colored("🔚 명령 처리 완료", Colors.GREEN)

    def _handle_continuous_mode(self):
        """연속 모드 처리 - 웨이크워드 없이 항상 명령어 대기"""
        print_colored("🎧 연속 모드 - 언제든지 명령어를 말씀하세요", Colors.GREEN + Colors.BOLD)

        while self.is_listening and config.continuous_mode:
            try:
                # 매번 새로운 스트림으로 명령어 듣기
                print_colored("🔊 명령어를 말씀하세요...", Colors.CYAN)

                cmd_stream, cmd_responses = self.start_stream(is_command_mode=False)
                last_transcript = ""

                for cmd_response in cmd_responses:
                    if not self.is_listening or not config.continuous_mode:
                        break

                    if not cmd_response.results:
                        continue

                    cmd_result = cmd_response.results[0]
                    if not cmd_result.alternatives:
                        continue

                    cmd_transcript = normalize(cmd_result.alternatives[0].transcript.strip())

                    if cmd_transcript and cmd_transcript != last_transcript:
                        # 실시간 표시
                        if cmd_result.is_final:
                            print_colored(f"  📝 인식: {cmd_transcript}", Colors.BLUE)
                        else:
                            print(f"\r  📝 듣는 중: {cmd_transcript}", end="", flush=True)

                        # 명령어 매칭 (final 결과에서만)
                        if cmd_result.is_final:
                            command = None
                            # 성능 최적화된 명령어 매칭
                            normalized_commands = get_normalized_commands()
                            for cmd, normalized_variations in normalized_commands.items():
                                if any(v in cmd_transcript for v in normalized_variations):
                                    command = cmd
                                    break

                            if command:
                                print()  # 줄바꿈
                                print_colored(f"✅ 명령어 인식: {command}", Colors.GREEN)

                                # 모드 변경 명령 체크
                                if command == "웨이크워드모드":
                                    print_colored("🔄 웨이크워드 모드로 전환합니다", Colors.CYAN)
                                    config.continuous_mode = False
                                    cmd_stream.__exit__(None, None, None)
                                    return
                                elif command == "연속모드":
                                    print_colored("✓ 이미 연속 모드입니다", Colors.GREEN)
                                    cmd_stream.__exit__(None, None, None)
                                    break

                                # 일반 명령 실행
                                if not self.execute_command(command):
                                    self.is_listening = False
                                    cmd_stream.__exit__(None, None, None)
                                    return

                                print_colored("🎧 계속 듣고 있습니다...", Colors.BLUE)
                                cmd_stream.__exit__(None, None, None)
                                break
                            else:
                                print()  # 줄바꿈
                                print_colored(f"  ❓ 알 수 없는 명령: {cmd_transcript}", Colors.WARNING)
                                # 잘못된 명령이어도 스트림을 종료하고 다시 시작
                                cmd_stream.__exit__(None, None, None)
                                break

                        last_transcript = cmd_transcript

                # 스트림 정리 (만약 아직 열려있다면)
                try:
                    cmd_stream.__exit__(None, None, None)
                except:
                    pass

                # 짧은 대기 후 다시 시작 (CPU 사용량 줄이기)
                if self.is_listening and config.continuous_mode:
                    time.sleep(0.1)

            except Exception as e:
                print_colored(f"❌ 연속 모드 오류: {e}", Colors.FAIL)
                time.sleep(1)  # 오류 발생 시 1초 대기 후 재시도

        print_colored("🔚 연속 모드 종료", Colors.GREEN)

# ===== 명령 처리 섹션 =====

class CommandProcessor:
    """명령 처리 클래스 - 코드 구조 개선"""

    def __init__(self):
        self.light_commands = {
            "조명켜": "ON",
            "조명꺼": "OFF",
            "밝게": "UP",
            "어둡게": "DOWN",
            "빨간색": "R",
            "파란색": "B",
            "녹색": "G",
            "노란색": "Y",
            "하얀색": "W",
            "무지개": "RAINBOW",
        }

        self.follow_commands = {
            "팔로우시작": start_hand_following,
            "팔로우정지": stop_hand_following,
            "초기화": reset_arm_position,
        }

        self.system_commands = {
            "아두이노연결": reconnect_arduino,
            "상태확인": check_arduino_status,
            "웨이크워드모드": switch_to_wake_word_mode,
            "연속모드": switch_to_continuous_mode,
        }

    def execute(self, command: str):
        """명령 실행"""
        if command in self.light_commands:
            print_colored(f"💡 조명 제어: {command}", Colors.CYAN)
            return send_arduino_command(self.light_commands[command])
        elif command in self.follow_commands:
            print_colored(f"🦾 로봇팔 제어: {command}", Colors.GREEN)
            self.follow_commands[command]()
            return True
        elif command in self.system_commands:
            print_colored(f"⚙️ 시스템 제어: {command}", Colors.BLUE)
            self.system_commands[command]()
            return True
        else:
            print_colored(f"❓ 알 수 없는 명령: {command}", Colors.WARNING)
            return False

# 전역 명령 처리기 인스턴스
_command_processor = CommandProcessor()

def robot_command_callback(command: str):
    """NeoPixel 조명 + 로봇팔 팔로우 명령 실행"""
    return _command_processor.execute(command)

# 전역 Arduino 터미널 인스턴스
_global_arduino = None

def get_arduino_terminal():
    """전역 Arduino 터미널 인스턴스 반환"""
    global _global_arduino
    if _global_arduino is None:
        _global_arduino = ArduinoTerminal()
        if SERIAL_AVAILABLE:
            try:
                _global_arduino.connect()
            except Exception as e:
                print_colored(f"⚠️ Arduino 초기 연결 실패: {e}", Colors.WARNING)
    return _global_arduino

def send_arduino_command(cmd: str):
    """Arduino에 NeoPixel 제어 명령 전송 (ArduinoTerminal 사용)"""
    if not SERIAL_AVAILABLE:
        print_colored(f"  ❌ pyserial 모듈이 설치되지 않음: pip install pyserial", Colors.FAIL)
        return False

    try:
        arduino = get_arduino_terminal()

        # 연결 상태 확인 및 재연결
        if not arduino.ser or not arduino.ser.is_open:
            print_colored(f"  🔄 Arduino 재연결 시도...", Colors.WARNING)
            if not arduino.connect():
                print_colored(f"  ❌ Arduino 연결 실패", Colors.FAIL)
                return False

        print_colored(f"  💡 Arduino 조명 제어: {cmd}", Colors.CYAN)
        return arduino.send_command(cmd)

    except Exception as e:
        print_colored(f"  ❌ Arduino 명령 전송 오류: {e}", Colors.FAIL)
        return False

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

def reconnect_arduino():
    """Arduino 재연결"""
    try:
        print_colored("  🔄 Arduino 재연결 시도", Colors.CYAN)
        arduino = get_arduino_terminal()

        # 기존 연결 해제
        if arduino.ser and arduino.ser.is_open:
            arduino.disconnect()

        # 재연결 시도
        if arduino.connect():
            print_colored("  ✓ Arduino 재연결 성공", Colors.GREEN)
        else:
            print_colored("  ❌ Arduino 재연결 실패", Colors.FAIL)

    except Exception as e:
        print_colored(f"  ❌ 재연결 오류: {e}", Colors.FAIL)

def check_arduino_status():
    """Arduino 상태 확인"""
    try:
        print_colored("  📊 Arduino 상태 확인", Colors.CYAN)
        arduino = get_arduino_terminal()

        if arduino.ser and arduino.ser.is_open:
            print_colored("  ✓ Arduino 연결 상태: 정상", Colors.GREEN)
            # 상태 명령 전송
            arduino.send_command("STATUS")
        else:
            print_colored("  ❌ Arduino 연결 상태: 끊어짐", Colors.FAIL)
            print_colored("  💡 '아두이노연결' 명령으로 재연결 가능", Colors.CYAN)

    except Exception as e:
        print_colored(f"  ❌ 상태 확인 오류: {e}", Colors.FAIL)

def switch_to_wake_word_mode():
    """웨이크워드 모드로 전환"""
    try:
        print_colored("  🔄 웨이크워드 모드로 전환", Colors.CYAN)
        config.continuous_mode = False
        print_colored("  ✓ 웨이크워드 모드 활성화 - '하이봇'을 말하고 명령하세요", Colors.GREEN)
    except Exception as e:
        print_colored(f"  ❌ 모드 전환 오류: {e}", Colors.FAIL)

def switch_to_continuous_mode():
    """연속 모드로 전환"""
    try:
        print_colored("  🔄 연속 모드로 전환", Colors.CYAN)
        config.continuous_mode = True
        print_colored("  ✓ 연속 모드 활성화 - 언제든지 명령어를 말씀하세요", Colors.GREEN)
    except Exception as e:
        print_colored(f"  ❌ 모드 전환 오류: {e}", Colors.FAIL)

def main():
    print_colored("🤖 통합 음성 제어 시스템 시작", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)

    # Arduino 초기 연결
    if SERIAL_AVAILABLE:
        arduino = get_arduino_terminal()
        if arduino.ser and arduino.ser.is_open:
            print_colored("✓ Arduino 준비 완료", Colors.GREEN)
        else:
            print_colored("⚠️ Arduino 연결 실패 - 음성 명령으로 재연결 시도 가능", Colors.WARNING)
    else:
        print_colored("⚠️ Arduino 제어 비활성화 (pyserial 모듈 없음)", Colors.WARNING)

    print_colored("=" * 50, Colors.HEADER)

    try:
        engine = VoiceRecognitionEngine()
        engine.set_command_callback(robot_command_callback)
        engine.run()
    finally:
        # 정리
        if _global_arduino:
            _global_arduino.disconnect()
        print_colored("👋 시스템 종료", Colors.GREEN)

if __name__ == "__main__":
    main()