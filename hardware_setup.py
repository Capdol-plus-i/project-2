#!/usr/bin/env python3
"""
Hardware Setup Tool for Robot Control System

Individual configuration tool for Arduino, Robot Arms, and Cameras.
"""

import sys
import os
import json
import cv2
import time
import threading
from utils.port_utils import get_recommended_ports, test_port_connection, find_arduino_ports, find_dynamixel_ports

class HardwareConfigManager:
    def __init__(self):
        self.config_file = 'hardware_config.json'
        self.load_config()
    
    def load_config(self):
        """Load hardware configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                # Default configuration
                self.config = {
                    'arduino': {
                        'enabled': True,
                        'port': 'COM5',
                        'baudrate': 9600,
                        'auto_detect': True
                    },
                    'robot_arms': {
                        'enabled': True,
                        'follower': {
                            'port': 'COM3',
                            'baudrate': 1000000,
                            'enabled': True
                        },
                        'leader': {
                            'port': 'COM4',
                            'baudrate': 1000000,
                            'enabled': True
                        },
                        'auto_detect': True
                    },
                    'cameras': {
                        'enabled': True,
                        'camera1': {
                            'id': 0,
                            'enabled': True
                        },
                        'camera2': {
                            'id': 2,
                            'enabled': True
                        },
                        'auto_detect': True
                    }
                }
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {}
    
    def save_config(self):
        """Save hardware configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print("OK Configuration saved to hardware_config.json")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def update_main_config(self):
        """Update the main config.py file with current settings"""
        try:
            # Read current config.py
            with open('config.py', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update relevant sections
            updated_lines = []
            in_arduino_section = False
            in_robot_section = False
            in_camera_section = False
            
            for line in lines:
                # Arduino configuration
                if "'port':" in line and "ARDUINO_CONFIG" in ''.join(updated_lines[-10:]):
                    updated_lines.append(f"    'port': '{self.config['arduino']['port']}',\n")
                elif "'enabled':" in line and "ARDUINO_CONFIG" in ''.join(updated_lines[-10:]):
                    updated_lines.append(f"    'enabled': {self.config['arduino']['enabled']},\n")
                
                # Robot arms configuration  
                elif "'port':" in line and "follower" in ''.join(updated_lines[-5:]):
                    updated_lines.append(f"        'port': '{self.config['robot_arms']['follower']['port']}',\n")
                elif "'port':" in line and "leader" in ''.join(updated_lines[-5:]):
                    updated_lines.append(f"        'port': '{self.config['robot_arms']['leader']['port']}',\n")
                
                # Camera configuration
                elif "'id':" in line and "camera1" in ''.join(updated_lines[-5:]):
                    updated_lines.append(f"        'id': {self.config['cameras']['camera1']['id']},\n")
                elif "'id':" in line and "camera2" in ''.join(updated_lines[-5:]):
                    updated_lines.append(f"        'id': {self.config['cameras']['camera2']['id']},\n")
                
                else:
                    updated_lines.append(line)
            
            # Write back to config.py
            with open('config.py', 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            print("OK Main config.py updated successfully!")
            return True
            
        except Exception as e:
            print(f"Error updating main config: {e}")
            return False

class ArduinoSetup:
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def show_current_config(self):
        """Display current Arduino configuration"""
        config = self.config_manager.config.get('arduino', {})
        print("\nCurrent Arduino Configuration:")
        print("=" * 40)
        print(f"Enabled:    {config.get('enabled', True)}")
        print(f"Port:       {config.get('port', 'Not set')}")
        print(f"Baudrate:   {config.get('baudrate', 9600)}")
        print(f"Auto-detect: {config.get('auto_detect', True)}")
    
    def auto_detect_arduino(self):
        """Auto-detect Arduino ports"""
        print("\nScanning for Arduino devices...")
        arduino_ports = find_arduino_ports()
        
        if arduino_ports:
            print("Found Arduino-like devices:")
            for i, port in enumerate(arduino_ports, 1):
                status = "OK" if test_port_connection(port) else "FAIL"
                print(f"{i}. {port} {status}")
            
            # Auto-select first available port
            for port in arduino_ports:
                if test_port_connection(port):
                    self.config_manager.config['arduino']['port'] = port
                    print(f"OK Auto-selected Arduino port: {port}")
                    return True
        else:
            print("No Arduino devices detected.")
            # Show all available ports as backup
            recommendations = get_recommended_ports()
            if recommendations['all_available']:
                print("Available serial ports:")
                for port in recommendations['all_available']:
                    status = "OK" if test_port_connection(port) else "FAIL"
                    print(f"  {port} {status}")
        
        return False
    
    def configure_arduino(self):
        """Interactive Arduino configuration"""
        print("\n" + "=" * 50)
        print("ARDUINO CONTROLLER SETUP")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Show current configuration")
            print("2. Auto-detect Arduino")
            print("3. Manually set port")
            print("4. Enable/disable Arduino")
            print("5. Set baudrate")
            print("6. Test connection")
            print("0. Back to main menu")
            
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.show_current_config()
            elif choice == '2':
                self.auto_detect_arduino()
            elif choice == '3':
                self.manual_port_setup()
            elif choice == '4':
                self.toggle_enable()
            elif choice == '5':
                self.set_baudrate()
            elif choice == '6':
                self.test_arduino_connection()
    
    def manual_port_setup(self):
        """Manual port configuration"""
        recommendations = get_recommended_ports()
        
        if recommendations['all_available']:
            print("\nAvailable ports:")
            for i, port in enumerate(recommendations['all_available'], 1):
                status = "OK" if test_port_connection(port) else "FAIL"
                print(f"{i}. {port} {status}")
            
            try:
                choice = input(f"\nSelect port (1-{len(recommendations['all_available'])}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(recommendations['all_available']):
                        port = recommendations['all_available'][idx]
                        self.config_manager.config['arduino']['port'] = port
                        print(f"OK Arduino port set to: {port}")
            except (ValueError, IndexError):
                print("Invalid selection.")
        else:
            # Manual entry
            port = input("Enter Arduino port (e.g., COM5, /dev/ttyACM2): ").strip()
            if port:
                self.config_manager.config['arduino']['port'] = port
                print(f"✓ Arduino port set to: {port}")
    
    def toggle_enable(self):
        """Toggle Arduino enable/disable"""
        current = self.config_manager.config['arduino'].get('enabled', True)
        new_state = not current
        self.config_manager.config['arduino']['enabled'] = new_state
        print(f"OK Arduino {'enabled' if new_state else 'disabled'}")
    
    def set_baudrate(self):
        """Set Arduino baudrate"""
        print("Common baudrates: 9600, 115200")
        try:
            baudrate = int(input("Enter baudrate (default 9600): ").strip() or "9600")
            self.config_manager.config['arduino']['baudrate'] = baudrate
            print(f"OK Arduino baudrate set to: {baudrate}")
        except ValueError:
            print("Invalid baudrate.")
    
    def test_arduino_connection(self):
        """Test Arduino connection"""
        config = self.config_manager.config['arduino']
        port = config.get('port')
        
        if not port:
            print("No Arduino port configured.")
            return
        
        print(f"Testing Arduino connection on {port}...")
        
        if test_port_connection(port, config.get('baudrate', 9600)):
            print("OK Arduino connection test successful!")
        else:
            print("FAIL Arduino connection test failed.")

class RobotArmsSetup:
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def show_current_config(self):
        """Display current robot arms configuration"""
        config = self.config_manager.config.get('robot_arms', {})
        print("\nCurrent Robot Arms Configuration:")
        print("=" * 40)
        print(f"Enabled:    {config.get('enabled', True)}")
        print(f"Auto-detect: {config.get('auto_detect', True)}")
        print("\nFollower Arm:")
        print(f"  Port:     {config.get('follower', {}).get('port', 'Not set')}")
        print(f"  Baudrate: {config.get('follower', {}).get('baudrate', 1000000)}")
        print(f"  Enabled:  {config.get('follower', {}).get('enabled', True)}")
        print("\nLeader Arm:")
        print(f"  Port:     {config.get('leader', {}).get('port', 'Not set')}")
        print(f"  Baudrate: {config.get('leader', {}).get('baudrate', 1000000)}")
        print(f"  Enabled:  {config.get('leader', {}).get('enabled', True)}")
    
    def auto_detect_robot_arms(self):
        """Auto-detect robot arm ports"""
        print("\nScanning for Dynamixel controllers...")
        dynamixel_ports = find_dynamixel_ports()
        
        if dynamixel_ports:
            print("Found Dynamixel-like devices:")
            for i, port in enumerate(dynamixel_ports, 1):
                status = "✓" if test_port_connection(port, 1000000) else "✗"
                print(f"{i}. {port} {status}")
            
            # Auto-assign ports
            if len(dynamixel_ports) >= 2:
                self.config_manager.config['robot_arms']['follower']['port'] = dynamixel_ports[0]
                self.config_manager.config['robot_arms']['leader']['port'] = dynamixel_ports[1]
                print(f"OK Auto-assigned Follower: {dynamixel_ports[0]}")
                print(f"OK Auto-assigned Leader: {dynamixel_ports[1]}")
            elif len(dynamixel_ports) == 1:
                self.config_manager.config['robot_arms']['follower']['port'] = dynamixel_ports[0]
                print(f"OK Auto-assigned Follower: {dynamixel_ports[0]}")
                print("! Only one Dynamixel controller found")
            return True
        else:
            print("No Dynamixel controllers detected.")
            return False
    
    def configure_robot_arms(self):
        """Interactive robot arms configuration"""
        print("\n" + "=" * 50)
        print("ROBOT ARMS SETUP")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Show current configuration")
            print("2. Auto-detect robot arms")
            print("3. Configure follower arm")
            print("4. Configure leader arm")
            print("5. Enable/disable robot arms")
            print("6. Test connections")
            print("0. Back to main menu")
            
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.show_current_config()
            elif choice == '2':
                self.auto_detect_robot_arms()
            elif choice == '3':
                self.configure_arm('follower')
            elif choice == '4':
                self.configure_arm('leader')
            elif choice == '5':
                self.toggle_enable()
            elif choice == '6':
                self.test_robot_connections()
    
    def configure_arm(self, arm_type):
        """Configure individual robot arm"""
        print(f"\nConfiguring {arm_type.title()} Arm:")
        
        recommendations = get_recommended_ports()
        if recommendations['all_available']:
            print("Available ports:")
            for i, port in enumerate(recommendations['all_available'], 1):
                status = "✓" if test_port_connection(port, 1000000) else "✗"
                print(f"{i}. {port} {status}")
            
            try:
                choice = input(f"Select port for {arm_type} arm (1-{len(recommendations['all_available'])}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(recommendations['all_available']):
                        port = recommendations['all_available'][idx]
                        self.config_manager.config['robot_arms'][arm_type]['port'] = port
                        print(f"OK {arm_type.title()} arm port set to: {port}")
            except (ValueError, IndexError):
                print("Invalid selection.")
    
    def toggle_enable(self):
        """Toggle robot arms enable/disable"""
        current = self.config_manager.config['robot_arms'].get('enabled', True)
        new_state = not current
        self.config_manager.config['robot_arms']['enabled'] = new_state
        print(f"OK Robot arms {'enabled' if new_state else 'disabled'}")
    
    def test_robot_connections(self):
        """Test robot arm connections"""
        config = self.config_manager.config['robot_arms']
        
        for arm_type in ['follower', 'leader']:
            port = config.get(arm_type, {}).get('port')
            if port:
                print(f"Testing {arm_type} arm connection on {port}...")
                if test_port_connection(port, 1000000):
                    print(f"OK {arm_type.title()} arm connection successful!")
                else:
                    print(f"FAIL {arm_type.title()} arm connection failed.")
            else:
                print(f"No port configured for {arm_type} arm.")

class CameraSetup:
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def show_current_config(self):
        """Display current camera configuration"""
        config = self.config_manager.config.get('cameras', {})
        print("\nCurrent Camera Configuration:")
        print("=" * 40)
        print(f"Enabled:    {config.get('enabled', True)}")
        print(f"Auto-detect: {config.get('auto_detect', True)}")
        print("\nCamera 1:")
        print(f"  ID:       {config.get('camera1', {}).get('id', 0)}")
        print(f"  Enabled:  {config.get('camera1', {}).get('enabled', True)}")
        print("\nCamera 2:")
        print(f"  ID:       {config.get('camera2', {}).get('id', 2)}")
        print(f"  Enabled:  {config.get('camera2', {}).get('enabled', True)}")
    
    def auto_detect_cameras(self):
        """Auto-detect available cameras"""
        print("\nScanning for cameras...")
        available_cameras = []
        
        # Test camera IDs 0-10
        for cam_id in range(11):
            try:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        available_cameras.append({'id': cam_id, 'resolution': f"{w}x{h}"})
                        print(f"OK Camera {cam_id}: {w}x{h}")
                    cap.release()
            except:
                pass
        
        if len(available_cameras) >= 2:
            self.config_manager.config['cameras']['camera1']['id'] = available_cameras[0]['id']
            self.config_manager.config['cameras']['camera2']['id'] = available_cameras[1]['id']
            print(f"OK Auto-assigned Camera 1: ID {available_cameras[0]['id']}")
            print(f"OK Auto-assigned Camera 2: ID {available_cameras[1]['id']}")
        elif len(available_cameras) == 1:
            self.config_manager.config['cameras']['camera1']['id'] = available_cameras[0]['id']
            self.config_manager.config['cameras']['camera2']['enabled'] = False
            print(f"OK Auto-assigned Camera 1: ID {available_cameras[0]['id']}")
            print("! Only one camera found, Camera 2 disabled")
        else:
            print("No cameras detected.")
        
        return len(available_cameras) > 0
    
    def preview_all_cameras(self):
        """Preview all available cameras with sample images"""
        print("\nScanning and capturing sample images from all cameras...")
        available_cameras = []
        sample_images = []
        
        # Test camera IDs 0-10 and capture sample images
        for cam_id in range(11):
            try:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    # Wait a bit for camera to initialize
                    time.sleep(0.5)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        available_cameras.append({'id': cam_id, 'resolution': f"{w}x{h}", 'frame': frame})
                        
                        # Save sample image
                        sample_filename = f"camera_{cam_id}_sample.jpg"
                        cv2.imwrite(sample_filename, frame)
                        sample_images.append(sample_filename)
                        print(f"OK Camera {cam_id}: {w}x{h} - Sample saved as {sample_filename}")
                    cap.release()
            except:
                pass
        
        if not available_cameras:
            print("No cameras detected.")
            return
        
        print(f"\nFound {len(available_cameras)} cameras:")
        for i, cam in enumerate(available_cameras):
            print(f"{i+1}. Camera ID {cam['id']}: {cam['resolution']}")
        
        # Show sample images info (GUI not available in this environment)
        print(f"\nSample images saved in current directory:")
        for filename in sample_images:
            print(f"  - {filename}")
        print("\nYou can open these image files to see what each camera captures.")
        
        # Ask user to select cameras
        try:
            print("\nSelect cameras for your configuration:")
            cam1_choice = input(f"Camera 1 (1-{len(available_cameras)}, or Enter to skip): ").strip()
            if cam1_choice and cam1_choice.isdigit():
                idx = int(cam1_choice) - 1
                if 0 <= idx < len(available_cameras):
                    cam_id = available_cameras[idx]['id']
                    self.config_manager.config['cameras']['camera1']['id'] = cam_id
                    print(f"OK Camera 1 set to ID {cam_id}")
            
            cam2_choice = input(f"Camera 2 (1-{len(available_cameras)}, or Enter to skip): ").strip()
            if cam2_choice and cam2_choice.isdigit():
                idx = int(cam2_choice) - 1
                if 0 <= idx < len(available_cameras):
                    cam_id = available_cameras[idx]['id']
                    self.config_manager.config['cameras']['camera2']['id'] = cam_id
                    print(f"OK Camera 2 set to ID {cam_id}")
        
        except (ValueError, IndexError):
            print("Invalid selection.")
        
        # Clean up sample images
        for filename in sample_images:
            try:
                os.remove(filename)
            except:
                pass
    
    def live_preview_camera(self):
        """Live preview of a specific camera (saves multiple snapshots)"""
        print("\nCamera Snapshot Preview (GUI not available)")
        print("=" * 50)
        
        # Show available cameras first
        available_cameras = []
        for cam_id in range(11):
            try:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        available_cameras.append({'id': cam_id, 'resolution': f"{w}x{h}"})
                    cap.release()
            except:
                pass
        
        if not available_cameras:
            print("No cameras detected.")
            return
        
        print("Available cameras:")
        for i, cam in enumerate(available_cameras):
            print(f"{i+1}. Camera ID {cam['id']}: {cam['resolution']}")
        
        try:
            choice = input(f"\nSelect camera for snapshot preview (1-{len(available_cameras)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_cameras):
                    cam_id = available_cameras[idx]['id']
                    self.capture_multiple_snapshots(cam_id)
        except (ValueError, IndexError):
            print("Invalid selection.")
    
    def capture_multiple_snapshots(self, cam_id):
        """Capture multiple snapshots from camera for preview"""
        print(f"\nCapturing multiple snapshots from Camera ID {cam_id}")
        
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Failed to open camera {cam_id}")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        snapshots = []
        print("Capturing 5 snapshots with 1 second intervals...")
        
        try:
            for i in range(5):
                # Wait a bit for camera to adjust
                time.sleep(1)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    snapshot_filename = f"camera_{cam_id}_preview_{i+1}.jpg"
                    cv2.imwrite(snapshot_filename, frame)
                    snapshots.append(snapshot_filename)
                    h, w = frame.shape[:2]
                    print(f"  Snapshot {i+1}: {snapshot_filename} ({w}x{h})")
                else:
                    print(f"  Failed to capture snapshot {i+1}")
        
        except Exception as e:
            print(f"Error during capture: {e}")
        
        finally:
            cap.release()
        
        if snapshots:
            print(f"\n{len(snapshots)} snapshots saved. You can open these files to see the camera output:")
            for filename in snapshots:
                print(f"  - {filename}")
            
            # Ask user if they want to assign this camera
            assign = input(f"\nAssign Camera {cam_id} to your configuration? (y/N): ").strip().lower()
            if assign.startswith('y'):
                print("1. Camera 1")
                print("2. Camera 2")
                choice = input("Select position (1-2): ").strip()
                if choice == '1':
                    self.config_manager.config['cameras']['camera1']['id'] = cam_id
                    print(f"OK Camera 1 assigned to ID {cam_id}")
                elif choice == '2':
                    self.config_manager.config['cameras']['camera2']['id'] = cam_id
                    print(f"OK Camera 2 assigned to ID {cam_id}")
            
            # Clean up snapshots after user sees them
            cleanup = input("Delete preview snapshots? (Y/n): ").strip().lower()
            if not cleanup.startswith('n'):
                for filename in snapshots:
                    try:
                        os.remove(filename)
                        print(f"Deleted {filename}")
                    except:
                        pass
    
    def configure_cameras(self):
        """Interactive camera configuration"""
        print("\n" + "=" * 50)
        print("CAMERA SETUP")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Show current configuration")
            print("2. Auto-detect cameras")
            print("3. Preview all cameras with sample images")
            print("4. Live preview camera")
            print("5. Set camera 1 ID")
            print("6. Set camera 2 ID")
            print("7. Enable/disable cameras")
            print("8. Test cameras")
            print("0. Back to main menu")
            
            choice = input("\nSelect option (0-8): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.show_current_config()
            elif choice == '2':
                self.auto_detect_cameras()
            elif choice == '3':
                self.preview_all_cameras()
            elif choice == '4':
                self.live_preview_camera()
            elif choice == '5':
                self.set_camera_id('camera1')
            elif choice == '6':
                self.set_camera_id('camera2')
            elif choice == '7':
                self.toggle_enable()
            elif choice == '8':
                self.test_cameras()
    
    def set_camera_id(self, camera_name):
        """Set camera ID manually"""
        try:
            cam_id = int(input(f"Enter camera ID for {camera_name} (0-10): ").strip())
            if 0 <= cam_id <= 10:
                self.config_manager.config['cameras'][camera_name]['id'] = cam_id
                print(f"OK {camera_name.title()} ID set to: {cam_id}")
            else:
                print("Camera ID must be between 0-10.")
        except ValueError:
            print("Invalid camera ID.")
    
    def toggle_enable(self):
        """Toggle cameras enable/disable"""
        current = self.config_manager.config['cameras'].get('enabled', True)
        new_state = not current
        self.config_manager.config['cameras']['enabled'] = new_state
        print(f"OK Cameras {'enabled' if new_state else 'disabled'}")
    
    def test_cameras(self):
        """Test camera connections"""
        config = self.config_manager.config['cameras']
        
        for camera_name in ['camera1', 'camera2']:
            cam_config = config.get(camera_name, {})
            if cam_config.get('enabled', True):
                cam_id = cam_config.get('id', 0)
                print(f"Testing {camera_name} (ID: {cam_id})...")
                
                try:
                    cap = cv2.VideoCapture(cam_id)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            print(f"OK {camera_name.title()} working: {w}x{h}")
                        else:
                            print(f"FAIL {camera_name.title()} opened but no frame")
                        cap.release()
                    else:
                        print(f"FAIL {camera_name.title()} failed to open")
                except Exception as e:
                    print(f"FAIL {camera_name.title()} error: {e}")

def main():
    """Main hardware setup tool"""
    print("Robot Control System - Hardware Setup Tool")
    print("=" * 60)
    
    config_manager = HardwareConfigManager()
    arduino_setup = ArduinoSetup(config_manager)
    robot_arms_setup = RobotArmsSetup(config_manager)
    camera_setup = CameraSetup(config_manager)
    
    while True:
        print("\nHardware Components:")
        print("1. Arduino Controller Setup")
        print("2. Robot Arms Setup")
        print("3. Camera Setup")
        print("4. Auto-detect All Hardware")
        print("5. Show All Configurations")
        print("6. Save Configuration")
        print("7. Update Main Config")
        print("8. Run Connection Test")
        print("0. Exit")
        
        try:
            choice = input("\nSelect component (0-8): ").strip()
            
            if choice == '0':
                # Ask if user wants to save before exiting
                if input("Save configuration before exit? (y/N): ").lower().startswith('y'):
                    config_manager.save_config()
                break
            elif choice == '1':
                arduino_setup.configure_arduino()
            elif choice == '2':
                robot_arms_setup.configure_robot_arms()
            elif choice == '3':
                camera_setup.configure_cameras()
            elif choice == '4':
                print("\nAuto-detecting all hardware...")
                arduino_setup.auto_detect_arduino()
                robot_arms_setup.auto_detect_robot_arms()
                camera_setup.auto_detect_cameras()
                print("OK Auto-detection complete!")
            elif choice == '5':
                arduino_setup.show_current_config()
                robot_arms_setup.show_current_config() 
                camera_setup.show_current_config()
            elif choice == '6':
                config_manager.save_config()
            elif choice == '7':
                if config_manager.update_main_config():
                    print("You can now run: python app.py")
            elif choice == '8':
                os.system("python test_connection_simple.py")
            else:
                print("Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            break
    
    print("\nHardware setup tool closed.")

if __name__ == "__main__":
    main()