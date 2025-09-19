#!/usr/bin/env python3
"""
Arduino 통신 테스트 스크립트
음성 인식 시스템에서 사용할 Arduino 명령들을 테스트합니다.
"""

import time
import serial
import sys

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

def test_arduino_connection():
    """Arduino 연결 테스트"""
    try:
        print_colored("🔌 Arduino 연결 테스트 시작...", Colors.CYAN)

        with serial.Serial('/dev/arduino', 9600, timeout=2) as ser:
            time.sleep(2)  # Arduino 부팅 대기

            print_colored("✓ Arduino 연결 성공!", Colors.GREEN)

            # 상태 요청
            print_colored("\n📊 Arduino 상태 확인...", Colors.BLUE)
            ser.write(b"STATUS\n")
            time.sleep(0.2)

            # 응답 읽기 (안전한 디코딩)
            while ser.in_waiting > 0:
                try:
                    raw_data = ser.readline()
                    response = raw_data.decode('utf-8', errors='ignore').strip()
                    if response:
                        clean_response = ''.join(char for char in response if ord(char) < 128)
                        if clean_response:
                            print_colored(f"  📨 {clean_response}", Colors.GREEN)
                except Exception:
                    pass

            return ser

    except serial.SerialException as e:
        print_colored(f"❌ Arduino 연결 실패: {e}", Colors.FAIL)
        print_colored("확인사항:", Colors.WARNING)
        print_colored("  - /dev/arduino 장치가 연결되어 있는지 확인", Colors.WARNING)
        print_colored("  - Arduino가 업로드되고 실행 중인지 확인", Colors.WARNING)
        print_colored("  - 다른 프로그램에서 시리얼 포트를 사용 중이지 않은지 확인", Colors.WARNING)
        return None
    except Exception as e:
        print_colored(f"❌ 예상치 못한 오류: {e}", Colors.FAIL)
        return None

def send_command(ser, command):
    """Arduino에 명령 전송"""
    try:
        print_colored(f"\n📡 명령 전송: {command}", Colors.BLUE)

        ser.write(f"{command}\n".encode('utf-8'))
        ser.flush()
        time.sleep(0.3)

        # 응답 읽기 (안전한 디코딩)
        responses = []
        while ser.in_waiting > 0:
            try:
                raw_data = ser.readline()
                response = raw_data.decode('utf-8', errors='ignore').strip()
                if response:
                    clean_response = ''.join(char for char in response if ord(char) < 128)
                    if clean_response:
                        responses.append(clean_response)
                        print_colored(f"  📨 {clean_response}", Colors.GREEN)
            except Exception:
                pass

        if not responses:
            print_colored("  (응답 없음)", Colors.WARNING)

        return responses

    except Exception as e:
        print_colored(f"❌ 명령 전송 실패: {e}", Colors.FAIL)
        return None

def test_voice_commands(ser):
    """음성 명령 테스트"""
    print_colored("\n🎤 음성 명령 테스트 시작", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)

    # 테스트할 명령들
    test_commands = [
        ("조명 켜기", "LIGHT_ON"),
        ("밝기 올리기", "BRIGHTNESS_UP"),
        ("밝기 올리기", "BRIGHTNESS_UP"),
        ("빨간색으로 변경", "COLOR_RED"),
        ("파란색으로 변경", "COLOR_BLUE"),
        ("녹색으로 변경", "COLOR_GREEN"),
        ("노란색으로 변경", "COLOR_YELLOW"),
        ("하얀색으로 변경", "COLOR_WHITE"),
        ("무지개 효과", "COLOR_RAINBOW"),
        ("밝기 내리기", "BRIGHTNESS_DOWN"),
        ("조명 끄기", "LIGHT_OFF"),
    ]

    for description, command in test_commands:
        print_colored(f"\n🔸 {description}", Colors.CYAN)
        send_command(ser, command)
        time.sleep(1)  # 효과 확인을 위한 딜레이

    print_colored("\n✅ 모든 음성 명령 테스트 완료!", Colors.GREEN)

def interactive_mode(ser):
    """대화형 모드"""
    print_colored("\n🎮 대화형 명령 모드", Colors.HEADER + Colors.BOLD)
    print_colored("사용 가능한 명령:", Colors.CYAN)
    print_colored("  LIGHT_ON, LIGHT_OFF", Colors.BLUE)
    print_colored("  BRIGHTNESS_UP, BRIGHTNESS_DOWN", Colors.BLUE)
    print_colored("  COLOR_RED, COLOR_BLUE, COLOR_GREEN", Colors.BLUE)
    print_colored("  COLOR_YELLOW, COLOR_WHITE, COLOR_RAINBOW", Colors.BLUE)
    print_colored("  STATUS, RESET", Colors.BLUE)
    print_colored("  'quit' 또는 'exit'로 종료", Colors.WARNING)
    print_colored("-" * 30, Colors.HEADER)

    while True:
        try:
            command = input(f"\n{Colors.CYAN}명령 입력: {Colors.ENDC}").strip()

            if command.lower() in ['quit', 'exit', 'q']:
                break

            if command:
                send_command(ser, command)

        except KeyboardInterrupt:
            print_colored("\n\n^C 사용자 중단", Colors.WARNING)
            break
        except EOFError:
            break

def main():
    print_colored("🤖 Arduino NeoPixel 통신 테스트", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)

    # Arduino 연결
    ser = test_arduino_connection()
    if not ser:
        sys.exit(1)

    try:
        # 음성 명령 테스트
        test_voice_commands(ser)

        # 대화형 모드
        interactive_mode(ser)

    except KeyboardInterrupt:
        print_colored("\n\n^C 프로그램 종료", Colors.WARNING)
    finally:
        if ser and ser.is_open:
            ser.close()
            print_colored("🔌 Arduino 연결 종료", Colors.GREEN)

if __name__ == "__main__":
    main()