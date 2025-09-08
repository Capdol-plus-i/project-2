#!/usr/bin/env python3
import os, time, threading, logging, signal, sys
from datetime import datetime

# Import modularized components
from config import MODEL_PATH, ARDUINO_PORT
from controllers.arduino_controller import ArduinoController
from controllers.voice_controller import VoiceController
from controllers.robot_controller import RobotController
from controllers.manipulator_robot import ManipulatorRobot
from utils.csv_utils import init_csv_file, save_snapshot_to_csv

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTerminalRobotControl:
    def __init__(self):
        self.robot = None
        self.controller = None
        self.arduino_controller = None
        self.voice_controller = None
        self.system_initialized = False
        self.running = True
        
    def print_header(self):
        """Print application header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("                        ROBOT CONTROL SYSTEM")
        print("=" * 80)
        
        # System status
        if self.system_initialized:
            print("✓ 시스템 상태: 준비 완료")
            if self.robot:
                connected_arms = list(self.robot.connected_arms)
                print(f"✓ 연결된 로봇 팔: {', '.join(connected_arms) if connected_arms else '없음'}")
            if self.arduino_controller and self.arduino_controller.is_connected():
                print("✓ 아두이노 상태: 연결됨")
            else:
                print("✗ 아두이노 상태: 연결 안됨")
        else:
            print("⏳ 시스템 상태: 초기화 중...")
        
        print("-" * 80)
    
    def show_main_menu(self):
        """Display main menu"""
        self.print_header()
        
        print("\n메인 메뉴:")
        print("1. 제스처 제어 시작/정지")
        print("2. 로봇 상태 확인")
        print("3. 아두이노 제어")
        print("4. 데이터 스냅샷 저장")
        print("5. 음성 제어 활성화/비활성화")
        print("6. 시스템 재시작")
        print("0. 종료")
        print("-" * 80)
        
        choice = input("선택하세요 (0-6): ").strip()
        return choice
    
    def show_robot_menu(self):
        """Display robot control menu"""
        self.print_header()
        
        print("\n로봇 제어 메뉴:")
        if self.controller:
            gesture_status = "활성화됨" if self.controller.control_active else "비활성화됨"
            print(f"현재 제스처 제어 상태: {gesture_status}")
        
        print("1. 제스처 제어 활성화")
        print("2. 제스처 제어 비활성화")
        print("3. 로봇 팔 토크 활성화")
        print("4. 로봇 팔 토크 비활성화")
        print("5. 로봇 상태 새로고침")
        print("0. 메인 메뉴로 돌아가기")
        print("-" * 80)
        
        choice = input("선택하세요 (0-5): ").strip()
        return choice
    
    def show_arduino_menu(self):
        """Display Arduino control menu"""
        self.print_header()
        
        print("\n아두이노 제어 메뉴:")
        if self.arduino_controller and self.arduino_controller.is_connected():
            brightness = self.arduino_controller.get_brightness()
            print(f"현재 밝기: {brightness}%")
        
        print("1. LED 밝게")
        print("2. LED 어둡게")
        print("3. LED 효과 (빨간색)")
        print("4. LED 효과 (초록색)")
        print("5. LED 효과 (파란색)")
        print("6. 아두이노 리셋")
        print("0. 메인 메뉴로 돌아가기")
        print("-" * 80)
        
        choice = input("선택하세요 (0-6): ").strip()
        return choice
    
    def init_system(self):
        """Initialize robot system"""
        try:
            print("시스템 초기화 중...")
            
            # Initialize Arduino controller
            print("Arduino 연결 중...")
            self.arduino_controller = ArduinoController(ARDUINO_PORT)
            if self.arduino_controller.connect():
                print("✓ Arduino controller 연결됨")
            else:
                print("✗ Arduino controller 연결 실패 - 계속 진행합니다")
            
            # Initialize robot
            print("로봇 연결 중...")
            self.robot = ManipulatorRobot()
            if not self.robot.connect():
                print("✗ 로봇 연결 실패 - 로봇 팔에 연결할 수 없습니다")
                return False

            # Initialize voice controller (optional component)
            try:
                print("음성 컨트롤러 초기화 중...")
                self.voice_controller = VoiceController()
                print("✓ 음성 컨트롤러 초기화됨")
            except Exception as e:
                print(f"✗ 음성 컨트롤러 초기화 실패: {e}")
                self.voice_controller = None

            # Update connected arms status
            connected_arms = list(self.robot.connected_arms)
            print(f"✓ 연결된 로봇 팔: {connected_arms}")
            
            # Setup robot arms
            print("로봇 팔 설정 중...")
            if self.robot.is_arm_connected('follower'):
                self.robot.disable_torque('follower')
            
            if self.robot.is_arm_connected('leader'):
                if not self.robot.setup_control('leader'):
                    print("⚠ Leader arm 설정 실패, 계속 진행합니다...")
            
            # Initialize controller
            print("카메라와 컨트롤러 시작 중...")
            self.controller = RobotController(self.robot, MODEL_PATH, 'follower')
            if not self.controller.start():
                self.robot.disconnect()
                if self.arduino_controller:
                    self.arduino_controller.disconnect()
                print("✗ 컨트롤러 시작 실패 - 카메라를 확인하세요")
                return False
            
            print("✓ 시스템 초기화 완료!")
            self.system_initialized = True
            
            # Arduino startup effect
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.trigger_led_effect(3)  # Green fade for successful init
            
            if self.voice_controller:
                self.voice_controller.start()

            return True
            
        except Exception as e:
            print(f"✗ 시스템 초기화 오류: {str(e)}")
            return False
    
    def cleanup_system(self):
        """Graceful system cleanup"""
        print("시스템 정리 중...")
        self.system_initialized = False
        
        if self.controller:
            try: 
                self.controller.stop()
            except Exception as e:
                logger.error(f"Controller cleanup error: {e}")
        if self.robot:
            try: 
                self.robot.disconnect()
            except Exception as e:
                logger.error(f"Robot cleanup error: {e}")
        if self.arduino_controller:
            try:
                self.arduino_controller.disconnect()
            except Exception as e:
                logger.error(f"Arduino cleanup error: {e}")
        if self.voice_controller:
            try:
                self.voice_controller.stop()
            except Exception as e:
                logger.error(f"Voice controller cleanup error: {e}")
        print("✓ 시스템 정리 완료")
    
    def handle_main_menu(self, choice):
        """Handle main menu choice"""
        if choice == '1':
            if not self.system_initialized:
                print("⚠ 시스템이 초기화되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                return
                
            if self.controller:
                if self.controller.control_active:
                    self.controller.stop_control()
                    print("✓ 제스처 제어가 비활성화되었습니다")
                else:
                    self.controller.start_control()
                    print("✓ 제스처 제어가 활성화되었습니다")
            else:
                print("⚠ 컨트롤러가 초기화되지 않았습니다")
            input("계속하려면 Enter를 누르세요...")
            
        elif choice == '2':
            self.robot_control_menu()
            
        elif choice == '3':
            self.arduino_control_menu()
            
        elif choice == '4':
            if self.robot and self.system_initialized:
                try:
                    positions = self.robot.get_current_positions()
                    success, count = save_snapshot_to_csv(positions)
                    if success:
                        print(f"✓ 데이터 스냅샷이 저장되었습니다 (총 {count}개)")
                    else:
                        print("✗ 스냅샷 저장 실패")
                except Exception as e:
                    print(f"✗ 스냅샷 저장 실패: {str(e)}")
            else:
                print("⚠ 로봇이 연결되지 않았습니다")
            input("계속하려면 Enter를 누르세요...")
            
        elif choice == '5':
            if self.voice_controller:
                print("✓ 음성 제어 상태 토글됨")
            else:
                print("⚠ 음성 컨트롤러가 초기화되지 않았습니다")
            input("계속하려면 Enter를 누르세요...")
            
        elif choice == '6':
            print("시스템을 재시작합니다...")
            self.cleanup_system()
            time.sleep(2)
            self.init_system()
            
        elif choice == '0':
            self.running = False
        else:
            print("⚠ 잘못된 선택입니다")
            input("계속하려면 Enter를 누르세요...")
    
    def robot_control_menu(self):
        """Robot control submenu"""
        while True:
            choice = self.show_robot_menu()
            
            if choice == '1':
                if self.controller:
                    self.controller.start_control()
                    print("✓ 제스처 제어가 활성화되었습니다")
                else:
                    print("⚠ 컨트롤러가 초기화되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '2':
                if self.controller:
                    self.controller.stop_control()
                    print("✓ 제스처 제어가 비활성화되었습니다")
                else:
                    print("⚠ 컨트롤러가 초기화되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '3':
                if self.robot:
                    self.robot.enable_torque('follower')
                    print("✓ 로봇 팔 토크가 활성화되었습니다")
                else:
                    print("⚠ 로봇이 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '4':
                if self.robot:
                    self.robot.disable_torque('follower')
                    print("✓ 로봇 팔 토크가 비활성화되었습니다")
                else:
                    print("⚠ 로봇이 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '5':
                print("✓ 로봇 상태를 새로고침했습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '0':
                break
            else:
                print("⚠ 잘못된 선택입니다")
                input("계속하려면 Enter를 누르세요...")
    
    def arduino_control_menu(self):
        """Arduino control submenu"""
        while True:
            choice = self.show_arduino_menu()
            
            if choice == '1':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.increase_brightness()
                    print("✓ LED가 밝아졌습니다")
                else:
                    print("⚠ 아두이노가 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '2':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.decrease_brightness()
                    print("✓ LED가 어두워졌습니다")
                else:
                    print("⚠ 아두이노가 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '3':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.trigger_led_effect(1)  # Red effect
                    print("✓ 빨간색 LED 효과 실행")
                else:
                    print("⚠ 아두이노가 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '4':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.trigger_led_effect(2)  # Green effect
                    print("✓ 초록색 LED 효과 실행")
                else:
                    print("⚠ 아두이노가 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '5':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.trigger_led_effect(3)  # Blue effect
                    print("✓ 파란색 LED 효과 실행")
                else:
                    print("⚠ 아두이노가 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '6':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.reset_arduino()
                    print("✓ 아두이노가 리셋되었습니다")
                else:
                    print("⚠ 아두이노가 연결되지 않았습니다")
                input("계속하려면 Enter를 누르세요...")
                
            elif choice == '0':
                break
            else:
                print("⚠ 잘못된 선택입니다")
                input("계속하려면 Enter를 누르세요...")
    
    def run(self):
        """Main application loop"""
        print("로봇 제어 시스템을 시작합니다...")
        
        # Initialize system
        if not self.init_system():
            print("시스템 초기화에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        try:
            while self.running:
                choice = self.show_main_menu()
                self.handle_main_menu(choice)
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다...")
        finally:
            self.cleanup_system()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info('Received shutdown signal, cleaning up...')
    sys.exit(0)

def main():
    """Main application entry point"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize CSV file
    init_csv_file()
    
    # Create and run terminal app
    app = SimpleTerminalRobotControl()
    app.run()

if __name__ == "__main__":
    main()