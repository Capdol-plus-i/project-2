#!/usr/bin/env python3
"""
Arduino 터미널 컨트롤러
터미널에서 직접 Arduino 명령을 입력하고 테스트할 수 있습니다.
"""

import time
import serial
import sys
import threading

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

class ArduinoTerminal:
    def __init__(self):
        self.ser = None
        self.listening = False
        self.listener_thread = None

    def connect(self):
        """Arduino 연결"""
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

    def start_listening(self):
        """Arduino 응답 실시간 수신"""
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen_responses, daemon=True)
        self.listener_thread.start()

    def _listen_responses(self):
        """Arduino 응답 수신 스레드 (안전한 디코딩)"""
        while self.listening and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    # 바이트 단위로 읽고 안전하게 디코딩
                    raw_data = self.ser.readline()
                    try:
                        # UTF-8 디코딩 시도
                        response = raw_data.decode('utf-8', errors='ignore').strip()
                    except UnicodeDecodeError:
                        # UTF-8 실패 시 Latin-1로 시도
                        response = raw_data.decode('latin-1', errors='ignore').strip()

                    if response and len(response) > 0:
                        # 비 ASCII 문자 제거
                        clean_response = ''.join(char for char in response if ord(char) < 128)
                        if clean_response:
                            print_colored(f"📨 Arduino: {clean_response}", Colors.GREEN)
                        else:
                            # 비 ASCII 데이터인 경우 hex로 표시
                            hex_data = ' '.join([f'{b:02x}' for b in raw_data[:20]])
                            print_colored(f"📨 Arduino (hex): {hex_data}", Colors.WARNING)
                time.sleep(0.1)
            except serial.SerialException as e:
                if self.listening:
                    print_colored(f"❌ 시리얼 연결 오류: {e}", Colors.FAIL)
                break
            except Exception as e:
                if self.listening:
                    print_colored(f"❌ 예상치 못한 수신 오류: {e}", Colors.FAIL)
                continue  # 에러 발생 시 break 대신 continue로 수정

    def send_command(self, command):
        """Arduino에 명령 전송"""
        if not self.ser or not self.ser.is_open:
            print_colored("❌ Arduino가 연결되지 않음", Colors.FAIL)
            return False

        try:
            print_colored(f"📡 전송: {command}", Colors.BLUE)
            self.ser.write(f"{command}\n".encode('utf-8'))
            self.ser.flush()
            return True

        except Exception as e:
            print_colored(f"❌ 전송 실패: {e}", Colors.FAIL)
            return False

    def disconnect(self):
        """Arduino 연결 종료"""
        self.listening = False
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1)

        if self.ser and self.ser.is_open:
            self.ser.close()
            print_colored("🔌 Arduino 연결 종료", Colors.GREEN)

def show_help():
    """도움말 표시"""
    print_colored("\n🎮 Arduino 명령어 목록:", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)

    print_colored("💡 조명 제어 (짧은 명령):", Colors.CYAN)
    print("  ON                - 조명 켜기")
    print("  OFF               - 조명 끄기")
    print("  UP                - 밝기 올리기")
    print("  DOWN              - 밝기 내리기")

    print_colored("\n🎨 색상 제어 (한 글자):", Colors.CYAN)
    print("  R                 - 빨간색")
    print("  B                 - 파란색")
    print("  G                 - 녹색")
    print("  Y                 - 노란색")
    print("  W                 - 하얀색")
    print("  RAINBOW           - 무지개 효과")

    print_colored("\n⚙️ 시스템:", Colors.CYAN)
    print("  STATUS            - 현재 상태 확인")
    print("  RESET             - 시스템 리셋")
    print("  HEARTBEAT         - 연결 확인")

    print_colored("\n🔧 고급 명령:", Colors.CYAN)
    print("  SET_BRIGHTNESS:N  - 밝기를 N단계로 설정 (0-5)")
    print("  SET_MODE:N        - 로봇 모드 설정 (0 또는 1)")
    print("  LED_EFFECT:N      - LED 효과 번호 (0-7)")

    print_colored("\n✨ 기존 긴 명령도 사용 가능:", Colors.WARNING)
    print("  LIGHT_ON, COLOR_RED 등도 지원됨")

    print_colored("\n📝 터미널 명령:", Colors.WARNING)
    print("  help, h           - 이 도움말 표시")
    print("  quit, q, exit     - 프로그램 종료")
    print("  clear, cls        - 화면 지우기")
    print_colored("=" * 50, Colors.HEADER)

def clear_screen():
    """화면 지우기"""
    import os
    os.system('clear' if os.name == 'posix' else 'cls')

def main():
    print_colored("🤖 Arduino 터미널 컨트롤러", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)

    terminal = ArduinoTerminal()

    # Arduino 연결
    if not terminal.connect():
        sys.exit(1)

    # 응답 수신 시작
    terminal.start_listening()

    # 초기 도움말
    show_help()

    print_colored("\n🎯 명령을 입력하세요 (help: 도움말, quit: 종료):", Colors.CYAN)

    try:
        while True:
            try:
                command = input(f"\n{Colors.CYAN}Arduino> {Colors.ENDC}").strip()

                if not command:
                    continue

                # 터미널 명령 처리
                if command.lower() in ['quit', 'q', 'exit']:
                    print_colored("👋 프로그램을 종료합니다.", Colors.WARNING)
                    break

                elif command.lower() in ['help', 'h']:
                    show_help()
                    continue

                elif command.lower() in ['clear', 'cls']:
                    clear_screen()
                    print_colored("🤖 Arduino 터미널 컨트롤러", Colors.HEADER + Colors.BOLD)
                    continue

                # Arduino 명령 전송
                terminal.send_command(command)

            except KeyboardInterrupt:
                print_colored("\n\n^C 감지됨. 'quit'를 입력하여 종료하세요.", Colors.WARNING)
                continue
            except EOFError:
                print_colored("\n\nEOF 감지됨. 프로그램을 종료합니다.", Colors.WARNING)
                break

    except Exception as e:
        print_colored(f"❌ 예상치 못한 오류: {e}", Colors.FAIL)
    finally:
        terminal.disconnect()

if __name__ == "__main__":
    main()