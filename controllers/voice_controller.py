#!/usr/bin/env python3
"""
Voice Controller module for speech recognition and voice command processing.
"""

import time
import threading
import logging
import pyaudio
from six.moves import queue
from google.cloud import speech
from config import VOICE_RATE, VOICE_CHUNK, WAKE_WORDS, VOICE_CMD_MAP

logger = logging.getLogger(__name__)

class MicrophoneStream:
    """참고 코드와 동일한 MicrophoneStream 클래스"""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._stream.stop_stream()
        self._stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

class VoiceController:
    def __init__(self):
        self.running = False
        self.listening = False
        self.current_stream = None
        self.client = None
        self.enabled = True  # 기본적으로 활성화
        # 음성 상태 추적
        self.voice_status = {
            'enabled': self.enabled,
            'listening': False,
            'last_transcript': '',
            'last_command': '',
            'wake_word_detected': False
        }

        try:
            self.client = speech.SpeechClient()
            logger.info("Voice controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice controller: {e}")
            self.enabled = False
    
    def start_stream(self, is_command_mode=False):
        """참고 코드와 동일한 스트림 생성 방식"""
        if not self.enabled:
            return None, None
            
        try:
            # 인식 우선순위 키워드 등록
            speech_context = speech.SpeechContext(
                phrases=WAKE_WORDS + [phrase for phrases in VOICE_CMD_MAP.values() for phrase in phrases],
                boost=20.0
            )

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=VOICE_RATE,
                language_code="ko-KR",
                speech_contexts=[speech_context],
                model="command_and_search"
            )

            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=is_command_mode  # 명령어 모드일 때는 말 끝나면 자동 종료
            )

            stream = MicrophoneStream(VOICE_RATE, VOICE_CHUNK)
            stream.__enter__()
            self.current_stream = stream
            
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
            responses = self.client.streaming_recognize(streaming_config, requests)
            
            return stream, responses
            
        except Exception as e:
            logger.error(f"Failed to create voice stream: {e}")
            return None, None
    
    def start(self):
        """음성 인식 시작"""
        if not self.enabled:
            logger.warning("Voice controller not available")
            return False
            
        if self.running:
            return True
            
        try:
            self.running = True
            self.listening = True
            self.voice_status['listening'] = True
            
            threading.Thread(target=self._voice_recognition_loop, daemon=True).start()
            logger.info("🎙️ Voice controller started - Say wake word to activate")
            
            # 웹 클라이언트에 상태 전송
            from app import socketio
            socketio.emit('voice_status_update', {
                'listening': True,
                'message': '웨이크워드를 말해주세요 (예: "하이봇")'
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to start voice controller: {e}")
            return False
    
    def stop(self):
        """음성 인식 중지"""
        self.running = False
        self.listening = False
        self.voice_status['listening'] = False
        
        if self.current_stream:
            try:
                self.current_stream.__exit__(None, None, None)
            except:
                pass
            self.current_stream = None
            
        from app import socketio
        socketio.emit('voice_status_update', {
            'listening': False,
            'message': 'Voice recognition stopped'
        })
        
        logger.info("Voice controller stopped")
    
    def _voice_recognition_loop(self):
        """참고 코드를 기반으로 한 음성 인식 메인 루프"""
        logger.info("🎙️ 시스템이 준비되었습니다. 웨이크워드를 말해주세요 (예: '하이봇')")

        while self.running:
            try:
                stream, responses = self.start_stream()
                if not stream or not responses:
                    time.sleep(1)
                    continue

                for response in responses:
                    if not self.running:
                        break
                        
                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript.strip()
                    logger.debug(f"인식된 내용: {transcript}")
                    
                    # 웹 클라이언트에 인식 내용 전송
                    from app import socketio
                    socketio.emit('voice_transcript', {'text': transcript})
                    
                    self.voice_status['last_transcript'] = transcript

                    # 웨이크워드 감지
                    if any(wake in transcript for wake in WAKE_WORDS):
                        if result.is_final or result.stability > 0.8:
                            logger.info(f"✅ 웨이크워드 감지됨: {transcript}")
                            self.voice_status['wake_word_detected'] = True
                            
                            # Arduino LED 효과
                            from app import arduino_controller
                            if arduino_controller and arduino_controller.is_connected():
                                arduino_controller.trigger_led_effect(2)  # 파란색 페이드
                            
                            # 웹 클라이언트에 웨이크워드 감지 알림
                            socketio.emit('voice_wake_word', {
                                'transcript': transcript,
                                'message': '웨이크워드 감지됨! 명령을 말씀하세요.'
                            })
                            
                            stream.__exit__(None, None, None)
                            time.sleep(0.1)
                            
                            # 명령어 모드 진입
                            self._command_mode()
                            break
            except Exception as e:
                logger.error(f"Voice recognition error: {e}")
                time.sleep(1)
    
    def _command_mode(self):
        """명령어 모드 - 참고 코드 패턴 적용"""
        try:
            logger.info("명령어 모드 진입")
            from app import socketio
            socketio.emit('voice_command_mode', {'active': True})
            
            # 명령어 모드 진입
            command_stream, command_responses = self.start_stream(is_command_mode=True)
            if not command_stream or not command_responses:
                return
                
            time.sleep(0.1)  # 스트림 안정화 대기
            start_time = time.time()
            MAX_COMMAND_DURATION = 5  # 초과 시 자동 종료

            for cmd_response in command_responses:
                if not self.running:
                    break
                    
                if time.time() - start_time > MAX_COMMAND_DURATION:
                    logger.info("명령어 대기 시간 초과")
                    socketio.emit('voice_timeout', {'message': '명령어 대기 시간 초과'})
                    command_stream.__exit__(None, None, None)
                    break

                if not cmd_response.results:
                    continue
                    
                cmd_result = cmd_response.results[0]
                if not cmd_result.alternatives:
                    continue

                cmd_transcript = cmd_result.alternatives[0].transcript.strip()
                logger.info(f"말한 내용: {cmd_transcript}")
                
                # 웹 클라이언트에 명령어 전송
                socketio.emit('voice_command_transcript', {'text': cmd_transcript})
                
                # 명령어 처리
                command_found = False
                for cmd, variations in VOICE_CMD_MAP.items():
                    if any(v in cmd_transcript for v in variations):
                        logger.info(f"✅ 명령어 실행: {cmd}")
                        self.voice_status['last_command'] = cmd
                        
                        # 명령어 실행
                        success = self._execute_command(cmd, cmd_transcript)
                        
                        # 웹 클라이언트에 명령어 실행 결과 전송
                        socketio.emit('voice_command_executed', {
                            'command': cmd,
                            'transcript': cmd_transcript,
                            'success': success
                        })
                        
                        command_stream.__exit__(None, None, None)
                        command_found = True
                        break
                        
                if command_found:
                    break
                    
        except Exception as e:
            logger.error(f"Command mode error: {e}")
        finally:
            from app import socketio
            socketio.emit('voice_command_mode', {'active': False})
    
    def _execute_command(self, cmd, transcript):
        """음성 명령어 실행"""
        try:
            from app import controller, arduino_controller, socketio
            
            if cmd == "시작":
                if controller and controller.running:
                    success = controller.start_control()
                    if success:
                        logger.info("✅ 제스처 제어 시작")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(3)  # 초록색 페이드
                    return success
                    
            elif cmd == "종료":
                if controller and controller.running:
                    success = controller.stop_control()
                    if success:
                        logger.info("✅ 제어 종료")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(1)  # 깜빡임
                    return success
            
            elif cmd == "정지":
                if controller and controller.running:
                    success = controller.pause_control()
                    if success:
                        logger.info("✅ 제스처 제어 정지")
                        if arduino_controller and arduino_controller.is_connected():
                            arduino_controller.trigger_led_effect(1)

            elif cmd == "밝게":
                if arduino_controller and arduino_controller.is_connected():
                    current_brightness = arduino_controller.get_status().get('brightness_level', 0)
                    new_brightness = min(5, current_brightness + 1)
                    success = arduino_controller.set_brightness(new_brightness)
                    if success:
                        logger.info(f"✅ 밝기 증가: {new_brightness}")
                    return success
                    
            elif cmd == "어둡게":
                if arduino_controller and arduino_controller.is_connected():
                    current_brightness = arduino_controller.get_status().get('brightness_level', 0)
                    new_brightness = max(0, current_brightness - 1)
                    success = arduino_controller.set_brightness(new_brightness)
                    if success:
                        logger.info(f"✅ 밝기 감소: {new_brightness}")
                    return success
                    
            elif cmd == "리셋":
                if arduino_controller and arduino_controller.is_connected():
                    success = arduino_controller.reset_arduino()
                    if success:
                        logger.info("✅ 아두이노 리셋")
                    return success

            elif cmd in ["왼쪽", "오른쪽", "위", "아래"]:
                logger.info(f"✅ 방향 명령: {cmd}")
                socketio.emit('voice_direction_command', {'direction': cmd})
                if arduino_controller and arduino_controller.is_connected():
                    arduino_controller.trigger_led_effect(1)  # 깜빡임
                return True
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False
            
        return False
    
    def get_status(self):
        """음성 컨트롤러 상태 반환"""
        return self.voice_status.copy()
    
    def is_enabled(self):
        """음성 컨트롤러 사용 가능 여부"""
        return self.enabled