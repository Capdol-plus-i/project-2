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
            print("[OK] System Status: Ready")
            if self.robot:
                connected_arms = list(self.robot.connected_arms)
                print(f"[OK] Connected Robot Arms: {', '.join(connected_arms) if connected_arms else 'None'}")
            if self.arduino_controller and self.arduino_controller.is_connected():
                print("[OK] Arduino Status: Connected")
            else:
                print("[X] Arduino Status: Disconnected")
        else:
            print("[...] System Status: Initializing...")
        
        print("-" * 80)
    
    def show_main_menu(self):
        """Display main menu"""
        self.print_header()
        
        print("\nMain Menu:")
        print("1. Toggle Gesture Control")
        print("2. Robot Control")
        print("3. Arduino Control")
        print("4. Save Data Snapshot")
        print("5. Voice Control Toggle")
        print("6. Restart System")
        print("0. Exit")
        print("-" * 80)
        
        choice = input("Select (0-6): ").strip()
        return choice
    
    def show_robot_menu(self):
        """Display robot control menu"""
        self.print_header()
        
        print("\nRobot Control Menu:")
        if self.controller:
            gesture_status = "Active" if self.controller.control_active else "Inactive"
            print(f"Current Gesture Control Status: {gesture_status}")
        
        print("1. Enable Gesture Control")
        print("2. Disable Gesture Control")
        print("3. Enable Robot Torque")
        print("4. Disable Robot Torque")
        print("5. Refresh Robot Status")
        print("0. Back to Main Menu")
        print("-" * 80)
        
        choice = input("Select (0-5): ").strip()
        return choice
    
    def show_arduino_menu(self):
        """Display Arduino control menu"""
        self.print_header()
        
        print("\nArduino Control Menu:")
        if self.arduino_controller and self.arduino_controller.is_connected():
            status = self.arduino_controller.get_status()
            brightness = status.get('brightness_level', 0)
            print(f"Current Brightness Level: {brightness}")
        
        print("1. LED Brighter")
        print("2. LED Dimmer")
        print("3. LED Effect (Red)")
        print("4. LED Effect (Green)")
        print("5. LED Effect (Blue)")
        print("6. Arduino Reset")
        print("0. Back to Main Menu")
        print("-" * 80)
        
        choice = input("Select (0-6): ").strip()
        return choice
    
    def init_system(self):
        """Initialize robot system"""
        try:
            print("Initializing system...")
            
            # Initialize Arduino controller
            print("Connecting to Arduino...")
            self.arduino_controller = ArduinoController(ARDUINO_PORT)
            if self.arduino_controller.connect():
                print("[OK] Arduino controller connected")
            else:
                print("[WARN] Arduino controller connection failed - continuing")
            
            # Initialize robot
            print("Connecting to robot...")
            self.robot = ManipulatorRobot()
            if not self.robot.connect():
                print("[ERROR] Robot connection failed - no robot arms could be connected")
                return False

            # Initialize voice controller (optional component)
            try:
                print("Initializing voice controller...")
                self.voice_controller = VoiceController()
                print("[OK] Voice controller initialized")
            except Exception as e:
                print(f"[WARN] Voice controller initialization failed: {e}")
                self.voice_controller = None

            # Update connected arms status
            connected_arms = list(self.robot.connected_arms)
            print(f"[OK] Connected robot arms: {connected_arms}")
            
            # Setup robot arms
            print("Setting up robot arms...")
            if self.robot.is_arm_connected('follower'):
                self.robot.disable_torque('follower')
            
            if self.robot.is_arm_connected('leader'):
                if not self.robot.setup_control('leader'):
                    print("[WARN] Leader arm setup failed, continuing...")
            
            # Initialize controller
            print("Starting cameras and controller...")
            self.controller = RobotController(self.robot, MODEL_PATH, 'follower')
            if not self.controller.start():
                self.robot.disconnect()
                if self.arduino_controller:
                    self.arduino_controller.disconnect()
                print("[ERROR] Controller start failed - check cameras")
                return False
            
            print("[OK] System initialization complete!")
            self.system_initialized = True
            
            # Arduino startup effect
            if self.arduino_controller and self.arduino_controller.is_connected():
                self.arduino_controller.trigger_led_effect(3)  # Green fade for successful init
            
            if self.voice_controller:
                self.voice_controller.start()

            return True
            
        except Exception as e:
            print(f"[ERROR] System initialization error: {str(e)}")
            return False
    
    def cleanup_system(self):
        """Graceful system cleanup"""
        print("Cleaning up system...")
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
        print("[OK] System cleanup complete")
    
    def handle_main_menu(self, choice):
        """Handle main menu choice"""
        if choice == '1':
            if not self.system_initialized:
                print("[WARN] System is not initialized")
                input("Press Enter to continue...")
                return
                
            if self.controller:
                if self.controller.control_active:
                    self.controller.stop_control()
                    print("[OK] Gesture control disabled")
                else:
                    self.controller.start_control()
                    print("[OK] Gesture control enabled")
            else:
                print("[WARN] Controller not initialized")
            input("Press Enter to continue...")
            
        elif choice == '2':
            self.robot_control_menu()
            
        elif choice == '3':
            self.arduino_control_menu()
            
        elif choice == '4':
            if self.robot and self.system_initialized:
                try:
                    positions = self.robot.get_positions('follower')
                    success, count = save_snapshot_to_csv(positions)
                    if success:
                        print(f"[OK] Data snapshot saved (Total: {count})")
                    else:
                        print("[ERROR] Snapshot save failed")
                except Exception as e:
                    print(f"[ERROR] Snapshot save error: {str(e)}")
            else:
                print("[WARN] Robot not connected")
            input("Press Enter to continue...")
            
        elif choice == '5':
            if self.voice_controller:
                print("[OK] Voice control toggled")
            else:
                print("[WARN] Voice controller not initialized")
            input("Press Enter to continue...")
            
        elif choice == '6':
            print("Restarting system...")
            self.cleanup_system()
            time.sleep(2)
            self.init_system()
            
        elif choice == '0':
            self.running = False
        else:
            print("[WARN] Invalid selection")
            input("Press Enter to continue...")
    
    def robot_control_menu(self):
        """Robot control submenu"""
        while True:
            choice = self.show_robot_menu()
            
            if choice == '1':
                if self.controller:
                    self.controller.start_control()
                    print("[OK] Gesture control enabled")
                else:
                    print("[WARN] Controller not initialized")
                input("Press Enter to continue...")
                
            elif choice == '2':
                if self.controller:
                    self.controller.stop_control()
                    print("[OK] Gesture control disabled")
                else:
                    print("[WARN] Controller not initialized")
                input("Press Enter to continue...")
                
            elif choice == '3':
                if self.robot:
                    self.robot.setup_control('follower')
                    print("[OK] Robot torque enabled")
                else:
                    print("[WARN] Robot not connected")
                input("Press Enter to continue...")
                
            elif choice == '4':
                if self.robot:
                    self.robot.disable_torque('follower')
                    print("[OK] Robot torque disabled")
                else:
                    print("[WARN] Robot not connected")
                input("Press Enter to continue...")
                
            elif choice == '5':
                print("[OK] Robot status refreshed")
                input("Press Enter to continue...")
                
            elif choice == '0':
                break
            else:
                print("[WARN] Invalid selection")
                input("Press Enter to continue...")
    
    def arduino_control_menu(self):
        """Arduino control submenu"""
        while True:
            choice = self.show_arduino_menu()
            
            if choice == '1':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    current_level = self.arduino_controller.get_status().get('brightness_level', 0)
                    new_level = min(5, current_level + 1)
                    self.arduino_controller.set_brightness(new_level)
                    print(f"[OK] LED brightness increased to level {new_level}")
                else:
                    print("[WARN] Arduino not connected")
                input("Press Enter to continue...")
                
            elif choice == '2':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    current_level = self.arduino_controller.get_status().get('brightness_level', 0)
                    new_level = max(0, current_level - 1)
                    self.arduino_controller.set_brightness(new_level)
                    print(f"[OK] LED brightness decreased to level {new_level}")
                else:
                    print("[WARN] Arduino not connected")
                input("Press Enter to continue...")
                
            elif choice == '3':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.trigger_led_effect(1)  # Red effect
                    print("[OK] Red LED effect executed")
                else:
                    print("[WARN] Arduino not connected")
                input("Press Enter to continue...")
                
            elif choice == '4':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.trigger_led_effect(2)  # Green effect
                    print("[OK] Green LED effect executed")
                else:
                    print("[WARN] Arduino not connected")
                input("Press Enter to continue...")
                
            elif choice == '5':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.trigger_led_effect(3)  # Blue effect
                    print("[OK] Blue LED effect executed")
                else:
                    print("[WARN] Arduino not connected")
                input("Press Enter to continue...")
                
            elif choice == '6':
                if self.arduino_controller and self.arduino_controller.is_connected():
                    self.arduino_controller.reset_arduino()
                    print("[OK] Arduino reset")
                else:
                    print("[WARN] Arduino not connected")
                input("Press Enter to continue...")
                
            elif choice == '0':
                break
            else:
                print("[WARN] Invalid selection")
                input("Press Enter to continue...")
    
    def run(self):
        """Main application loop"""
        print("Starting Robot Control System...")
        
        # Initialize system
        if not self.init_system():
            print("System initialization failed. Exiting.")
            return
        
        try:
            while self.running:
                choice = self.show_main_menu()
                self.handle_main_menu(choice)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
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