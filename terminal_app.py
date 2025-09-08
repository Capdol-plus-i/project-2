#!/usr/bin/env python3
import os, time, threading, logging, signal, sys
import curses
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

class TerminalRobotControl:
    def __init__(self):
        self.robot = None
        self.controller = None
        self.arduino_controller = None
        self.voice_controller = None
        self.system_initialized = False
        self.running = True
        self.current_menu = 'main'
        
        # Status tracking
        self.system_status = {'status': 'initializing', 'message': 'System starting...'}
        self.robot_status = {'connected_arms': []}
        
    def init_curses(self, stdscr):
        """Initialize curses interface"""
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
        
    def display_header(self):
        """Display application header"""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "=" * 80, curses.color_pair(4))
        self.stdscr.addstr(1, 30, "ROBOT CONTROL SYSTEM", curses.color_pair(4) | curses.A_BOLD)
        self.stdscr.addstr(2, 0, "=" * 80, curses.color_pair(4))
        
        # System status
        status_color = curses.color_pair(1) if self.system_status['status'] == 'ready' else curses.color_pair(2)
        self.stdscr.addstr(4, 0, f"시스템 상태: {self.system_status['status'].upper()}", status_color)
        self.stdscr.addstr(5, 0, f"메시지: {self.system_status['message']}")
        
        # Robot status
        if self.robot_status['connected_arms']:
            self.stdscr.addstr(6, 0, f"연결된 로봇 팔: {', '.join(self.robot_status['connected_arms'])}", curses.color_pair(1))
        
        # Arduino status
        if self.arduino_controller:
            arduino_status = "연결됨" if self.arduino_controller.is_connected() else "연결 안됨"
            arduino_color = curses.color_pair(1) if self.arduino_controller.is_connected() else curses.color_pair(2)
            self.stdscr.addstr(7, 0, f"아두이노 상태: {arduino_status}", arduino_color)
        
        self.stdscr.addstr(9, 0, "-" * 80)
        
    def display_main_menu(self):
        """Display main menu"""
        self.display_header()
        
        menu_items = [
            "1. 제스처 제어 시작/정지",
            "2. 로봇 상태 확인",
            "3. 아두이노 제어",
            "4. 데이터 스냅샷 저장",
            "5. 시스템 설정",
            "6. 음성 제어 활성화/비활성화",
            "7. 시스템 재시작",
            "q. 종료"
        ]
        
        self.stdscr.addstr(11, 0, "메인 메뉴:", curses.A_BOLD)
        for i, item in enumerate(menu_items):
            self.stdscr.addstr(13 + i, 2, item)
        
        self.stdscr.addstr(22, 0, "선택하세요: ")
        
    def display_robot_menu(self):
        """Display robot control menu"""
        self.display_header()
        
        self.stdscr.addstr(11, 0, "로봇 제어 메뉴:", curses.A_BOLD)
        
        if self.controller:
            gesture_status = "활성화됨" if self.controller.control_active else "비활성화됨"
            self.stdscr.addstr(13, 2, f"제스처 제어 상태: {gesture_status}")
        
        menu_items = [
            "1. 제스처 제어 활성화",
            "2. 제스처 제어 비활성화", 
            "3. 로봇 팔 토크 활성화",
            "4. 로봇 팔 토크 비활성화",
            "5. 로봇 상태 새로고침",
            "b. 메인 메뉴로 돌아가기"
        ]
        
        for i, item in enumerate(menu_items):
            self.stdscr.addstr(15 + i, 2, item)
        
        self.stdscr.addstr(22, 0, "선택하세요: ")
        
    def display_arduino_menu(self):
        """Display Arduino control menu"""
        self.display_header()
        
        self.stdscr.addstr(11, 0, "아두이노 제어 메뉴:", curses.A_BOLD)
        
        if self.arduino_controller:
            brightness = self.arduino_controller.get_brightness()
            self.stdscr.addstr(13, 2, f"현재 밝기: {brightness}%")
        
        menu_items = [
            "1. LED 밝게",
            "2. LED 어둡게", 
            "3. LED 효과 (빨간색)",
            "4. LED 효과 (초록색)",
            "5. LED 효과 (파란색)",
            "6. 아두이노 리셋",
            "b. 메인 메뉴로 돌아가기"
        ]
        
        for i, item in enumerate(menu_items):
            self.stdscr.addstr(15 + i, 2, item)
        
        self.stdscr.addstr(22, 0, "선택하세요: ")
    
    def init_system(self):
        """Initialize robot system"""
        try:
            logger.info("Initializing robot system...")
            self.system_status = {'status': 'initializing', 'message': 'Arduino 연결 중...'}
            
            # Initialize Arduino controller
            self.arduino_controller = ArduinoController(ARDUINO_PORT)
            if self.arduino_controller.connect():
                logger.info("Arduino controller connected")
            else:
                logger.warning("Arduino controller not connected - continuing without it")
            
            self.system_status = {'status': 'initializing', 'message': '로봇 연결 중...'}
            
            # Initialize robot
            self.robot = ManipulatorRobot()
            if not self.robot.connect():
                error_msg = '로봇 연결 실패 - 로봇 팔에 연결할 수 없습니다'
                logger.error(error_msg)
                self.system_status = {'status': 'error', 'message': error_msg}
                return False

            # Initialize voice controller (optional component)
            try:
                self.system_status = {'status': 'initializing', 'message': '음성 컨트롤러 초기화 중...'}
                self.voice_controller = VoiceController()
                logger.info("Voice controller initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize voice controller: {e}")
                self.voice_controller = None

            # Update connected arms status
            connected_arms = list(self.robot.connected_arms)
            self.robot_status = {'connected_arms': connected_arms}
            logger.info(f"Connected robot arms: {connected_arms}")
            
            self.system_status = {'status': 'initializing', 'message': '로봇 팔 설정 중...'}
            
            # Setup robot arms
            if self.robot.is_arm_connected('follower'):
                self.robot.disable_torque('follower')
            
            if self.robot.is_arm_connected('leader'):
                if not self.robot.setup_control('leader'):
                    logger.warning('Leader arm setup failed, but continuing...')
            
            self.system_status = {'status': 'initializing', 'message': '카메라와 컨트롤러 시작 중...'}
            
            # Initialize controller
            self.controller = RobotController(self.robot, MODEL_PATH, 'follower')
            if not self.controller.start():
                self.robot.disconnect()
                if self.arduino_controller:
                    self.arduino_controller.disconnect()
                error_msg = '컨트롤러 시작 실패 - 카메라를 확인하세요'
                logger.error(error_msg)
                self.system_status = {'status': 'error', 'message': error_msg}
                return False
            
            logger.info("System initialization complete!")
            status_message = f'시스템 준비 완료 - 로봇: {connected_arms}, 아두이노: {"연결됨" if self.arduino_controller and self.arduino_controller.is_connected() else "연결 안됨"}'
            self.system_status = {'status': 'ready', 'message': status_message}
            self.system_initialized = True
            
            # Arduino startup effect
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.trigger_led_effect(3)  # Green fade for successful init
            
            if self.voice_controller:
                self.voice_controller.start()

            return True
            
        except Exception as e:
            error_msg = f'시스템 초기화 오류: {str(e)}'
            logger.error(error_msg)
            self.system_status = {'status': 'error', 'message': error_msg}
            return False
    
    def cleanup_system(self):
        """Graceful system cleanup"""
        logger.info("Cleaning up system...")
        self.system_initialized = False
        self.system_status = {'status': 'error', 'message': '시스템 종료 중'}
        self.robot_status = {'connected_arms': []}
        
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
        logger.info("System cleanup complete")
    
    def handle_main_menu(self, key):
        """Handle main menu input"""
        if key == '1':
            if self.controller:
                if self.controller.control_active:
                    self.controller.stop_control()
                    self.show_message("제스처 제어가 비활성화되었습니다")
                else:
                    self.controller.start_control()
                    self.show_message("제스처 제어가 활성화되었습니다")
            else:
                self.show_message("컨트롤러가 초기화되지 않았습니다")
        elif key == '2':
            self.current_menu = 'robot'
        elif key == '3':
            self.current_menu = 'arduino'
        elif key == '4':
            if self.robot and self.system_initialized:
                try:
                    positions = self.robot.get_positions('follower')
                    success, count = save_snapshot_to_csv(positions)
                    if success:
                        self.show_message(f"데이터 스냅샷이 저장되었습니다 (총 {count}개)")
                    else:
                        self.show_message("스냅샷 저장 실패")
                except Exception as e:
                    self.show_message(f"스냅샷 저장 실패: {str(e)}")
            else:
                self.show_message("로봇이 연결되지 않았습니다")
        elif key == '5':
            self.show_message("설정 메뉴는 아직 구현되지 않았습니다")
        elif key == '6':
            if self.voice_controller:
                self.show_message("음성 제어 토글됨")
            else:
                self.show_message("음성 컨트롤러가 초기화되지 않았습니다")
        elif key == '7':
            self.restart_system()
        elif key == 'q':
            self.running = False
    
    def handle_robot_menu(self, key):
        """Handle robot menu input"""
        if key == '1':
            if self.controller:
                self.controller.start_control()
                self.show_message("제스처 제어가 활성화되었습니다")
        elif key == '2':
            if self.controller:
                self.controller.stop_control()
                self.show_message("제스처 제어가 비활성화되었습니다")
        elif key == '3':
            if self.robot:
                self.robot.setup_control('follower')
                self.show_message("로봇 팔 토크가 활성화되었습니다")
        elif key == '4':
            if self.robot:
                self.robot.disable_torque('follower')
                self.show_message("로봇 팔 토크가 비활성화되었습니다")
        elif key == '5':
            self.show_message("로봇 상태를 새로고침했습니다")
        elif key == 'b':
            self.current_menu = 'main'
    
    def handle_arduino_menu(self, key):
        """Handle Arduino menu input"""
        if key == '1':
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.increase_brightness()
                self.show_message("LED가 밝아졌습니다")
        elif key == '2':
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.decrease_brightness()
                self.show_message("LED가 어두워졌습니다")
        elif key == '3':
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.trigger_led_effect(1)  # Red effect
                self.show_message("빨간색 LED 효과 실행")
        elif key == '4':
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.trigger_led_effect(2)  # Green effect
                self.show_message("초록색 LED 효과 실행")
        elif key == '5':
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.trigger_led_effect(3)  # Blue effect
                self.show_message("파란색 LED 효과 실행")
        elif key == '6':
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.reset_arduino()
                self.show_message("아두이노가 리셋되었습니다")
        elif key == 'b':
            self.current_menu = 'main'
    
    def show_message(self, message):
        """Show temporary message"""
        self.stdscr.addstr(24, 0, " " * 80)  # Clear line
        self.stdscr.addstr(24, 0, f"메시지: {message}", curses.color_pair(3))
        self.stdscr.refresh()
        time.sleep(2)
    
    def restart_system(self):
        """Restart the system"""
        self.show_message("시스템을 재시작하는 중...")
        self.cleanup_system()
        threading.Thread(target=self.init_system, daemon=True).start()
    
    def run(self, stdscr):
        """Main application loop"""
        self.init_curses(stdscr)
        
        # Initialize system in background
        threading.Thread(target=self.init_system, daemon=True).start()
        
        while self.running:
            try:
                if self.current_menu == 'main':
                    self.display_main_menu()
                elif self.current_menu == 'robot':
                    self.display_robot_menu()
                elif self.current_menu == 'arduino':
                    self.display_arduino_menu()
                
                self.stdscr.refresh()
                
                # Get input
                key = self.stdscr.getch()
                if key != -1:  # Key was pressed
                    key_str = chr(key) if 32 <= key <= 126 else None
                    
                    if key_str:
                        if self.current_menu == 'main':
                            self.handle_main_menu(key_str)
                        elif self.current_menu == 'robot':
                            self.handle_robot_menu(key_str)
                        elif self.current_menu == 'arduino':
                            self.handle_arduino_menu(key_str)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except KeyboardInterrupt:
                break
        
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
    app = TerminalRobotControl()
    
    try:
        curses.wrapper(app.run)
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        app.cleanup_system()

if __name__ == "__main__":
    main()