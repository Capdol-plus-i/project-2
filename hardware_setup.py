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
import glob
import serial.tools.list_ports
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
                        'port': '/dev/ttyACM0',
                        'baudrate': 9600,
                        'auto_detect': True
                    },
                    'robot_arms': {
                        'enabled': True,
                        'follower': {
                            'port': '/dev/ttyACM1',
                            'baudrate': 1000000,
                            'enabled': True
                        },
                        'leader': {
                            'port': '/dev/ttyACM2',
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
            print("✓ Configuration saved to hardware_config.json")
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
            
            print("✓ Main config.py updated successfully!")
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
        print(f"Enabled:     {config.get('enabled', True)}")
        print(f"Port:        {config.get('port', 'Not set')}")
        print(f"Baudrate:    {config.get('baudrate', 9600)}")
        print(f"Auto-detect: {config.get('auto_detect', True)}")
    
    def auto_detect_arduino(self):
        """Auto-detect Arduino ports"""
        print("\n🔍 Scanning for Arduino devices...")
        
        # Get all serial ports
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        # Look for Arduino-like devices
        for port in ports:
            description = port.description.lower()
            manufacturer = (port.manufacturer or "").lower()
            
            if any(keyword in description for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                arduino_ports.append(port.device)
                print(f"Found Arduino-like device: {port.device} ({port.description})")
            elif any(keyword in manufacturer for keyword in ['arduino', 'ch340', 'silicon labs', 'ftdi']):
                arduino_ports.append(port.device)
                print(f"Found Arduino-like device: {port.device} ({manufacturer})")
        
        if arduino_ports:
            print("Testing connections...")
            for port in arduino_ports:
                status = "✓" if test_port_connection(port) else "✗"
                print(f"  {port} {status}")
            
            # Auto-select first working port
            for port in arduino_ports:
                if test_port_connection(port):
                    self.config_manager.config['arduino']['port'] = port
                    print(f"✓ Auto-selected Arduino port: {port}")
                    return True
        else:
            print("No Arduino devices detected.")
            print("Available serial ports:")
            for port in ports:
                status = "✓" if test_port_connection(port.device) else "✗"
                print(f"  {port.device} - {port.description} {status}")
        
        return False
    
    def configure_arduino(self):
        """Interactive Arduino configuration"""
        print("\n" + "=" * 50)
        print("🔧 ARDUINO CONTROLLER SETUP")
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
        ports = serial.tools.list_ports.comports()
        
        if ports:
            print("\nAvailable ports:")
            for i, port in enumerate(ports, 1):
                status = "✓" if test_port_connection(port.device) else "✗"
                print(f"{i}. {port.device} - {port.description} {status}")
            
            try:
                choice = input(f"\nSelect port (1-{len(ports)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(ports):
                        port = ports[idx].device
                        self.config_manager.config['arduino']['port'] = port
                        print(f"✓ Arduino port set to: {port}")
            except (ValueError, IndexError):
                print("Invalid selection.")
        else:
            # Manual entry for Linux
            port = input("Enter Arduino port (e.g., /dev/ttyACM0, /dev/ttyUSB0): ").strip()
            if port:
                self.config_manager.config['arduino']['port'] = port
                print(f"✓ Arduino port set to: {port}")
    
    def toggle_enable(self):
        """Toggle Arduino enable/disable"""
        current = self.config_manager.config['arduino'].get('enabled', True)
        new_state = not current
        self.config_manager.config['arduino']['enabled'] = new_state
        print(f"✓ Arduino {'enabled' if new_state else 'disabled'}")
    
    def set_baudrate(self):
        """Set Arduino baudrate"""
        print("Common baudrates: 9600, 57600, 115200")
        try:
            baudrate = int(input("Enter baudrate (default 9600): ").strip() or "9600")
            self.config_manager.config['arduino']['baudrate'] = baudrate
            print(f"✓ Arduino baudrate set to: {baudrate}")
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
            print("✓ Arduino connection test successful!")
        else:
            print("✗ Arduino connection test failed.")

class RobotArmsSetup:
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def show_current_config(self):
        """Display current robot arms configuration"""
        config = self.config_manager.config.get('robot_arms', {})
        print("\nCurrent Robot Arms Configuration:")
        print("=" * 40)
        print(f"Enabled:     {config.get('enabled', True)}")
        print(f"Auto-detect: {config.get('auto_detect', True)}")
        print("\nFollower Arm:")
        print(f"  Port:      {config.get('follower', {}).get('port', 'Not set')}")
        print(f"  Baudrate:  {config.get('follower', {}).get('baudrate', 1000000)}")
        print(f"  Enabled:   {config.get('follower', {}).get('enabled', True)}")
        print("\nLeader Arm:")
        print(f"  Port:      {config.get('leader', {}).get('port', 'Not set')}")
        print(f"  Baudrate:  {config.get('leader', {}).get('baudrate', 1000000)}")
        print(f"  Enabled:   {config.get('leader', {}).get('enabled', True)}")
    
    def auto_detect_robot_arms(self):
        """Auto-detect robot arm ports - very lenient"""
        print("\n🔍 Scanning for robot arm controllers...")
        
        ports = serial.tools.list_ports.comports()
        potential_ports = []
        
        print(f"Found {len(ports)} serial ports:")
        for port in ports:
            description = port.description.lower()
            manufacturer = (port.manufacturer or "").lower()
            
            print(f"  {port.device}: {port.description}")
            
            # Very lenient detection - any serial device could be robot arms
            if any(keyword in description for keyword in ['usb', 'serial', 'acm', 'arduino', 'ch340', 'cp210', 'ftdi']) or port.device.startswith('/dev/ttyACM') or port.device.startswith('/dev/ttyUSB'):
                potential_ports.append(port.device)
                print(f"    → Potential robot arm port")
        
        if potential_ports:
            print(f"\nTesting {len(potential_ports)} potential robot arm ports...")
            working_ports = []
            
            for port in potential_ports:
                print(f"Testing {port}...", end=" ")
                
                # Very lenient test - just check if port can be opened
                import serial as pyserial
                try:
                    # Try to open the port with default Dynamixel baudrate
                    with pyserial.Serial(port, 1000000, timeout=0.1) as ser:
                        print(f"✓ Port opens (1000000 baud)")
                        working_ports.append({'port': port, 'baudrate': 1000000})
                except pyserial.SerialException:
                    try:
                        # Try with lower baudrate
                        with pyserial.Serial(port, 57600, timeout=0.1) as ser:
                            print(f"✓ Port opens (57600 baud)")
                            working_ports.append({'port': port, 'baudrate': 57600})
                    except pyserial.SerialException:
                        try:
                            # Try with standard baudrate
                            with pyserial.Serial(port, 9600, timeout=0.1) as ser:
                                print(f"✓ Port opens (9600 baud)")
                                working_ports.append({'port': port, 'baudrate': 9600})
                        except pyserial.SerialException:
                            print("✗ Cannot open port")
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            # Auto-assign ports
            if len(working_ports) >= 2:
                self.config_manager.config['robot_arms']['follower']['port'] = working_ports[0]['port']
                self.config_manager.config['robot_arms']['follower']['baudrate'] = working_ports[0]['baudrate']
                self.config_manager.config['robot_arms']['leader']['port'] = working_ports[1]['port']
                self.config_manager.config['robot_arms']['leader']['baudrate'] = working_ports[1]['baudrate']
                print(f"✓ Auto-assigned Follower: {working_ports[0]['port']} @ {working_ports[0]['baudrate']}")
                print(f"✓ Auto-assigned Leader: {working_ports[1]['port']} @ {working_ports[1]['baudrate']}")
                return True
            elif len(working_ports) == 1:
                self.config_manager.config['robot_arms']['follower']['port'] = working_ports[0]['port']
                self.config_manager.config['robot_arms']['follower']['baudrate'] = working_ports[0]['baudrate']
                self.config_manager.config['robot_arms']['leader']['port'] = working_ports[0]['port']
                self.config_manager.config['robot_arms']['leader']['baudrate'] = working_ports[0]['baudrate']
                print(f"✓ Auto-assigned both arms to: {working_ports[0]['port']} @ {working_ports[0]['baudrate']}")
                print("ℹ Both arms using same port (single controller setup)")
                return True
            else:
                print("No working robot arm controllers found.")
                return False
        else:
            print("No potential robot arm ports detected.")
            return False
    
    def configure_robot_arms(self):
        """Interactive robot arms configuration"""
        print("\n" + "=" * 50)
        print("🦾 ROBOT ARMS SETUP")
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
        print(f"\n⚙️ Configuring {arm_type.title()} Arm:")
        
        ports = serial.tools.list_ports.comports()
        if ports:
            print("Available ports:")
            for i, port in enumerate(ports, 1):
                status = "✓" if test_port_connection(port.device, 1000000) else "✗"
                print(f"{i}. {port.device} - {port.description} {status}")
            
            try:
                choice = input(f"Select port for {arm_type} arm (1-{len(ports)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(ports):
                        port = ports[idx].device
                        self.config_manager.config['robot_arms'][arm_type]['port'] = port
                        print(f"✓ {arm_type.title()} arm port set to: {port}")
            except (ValueError, IndexError):
                print("Invalid selection.")
    
    def toggle_enable(self):
        """Toggle robot arms enable/disable"""
        current = self.config_manager.config['robot_arms'].get('enabled', True)
        new_state = not current
        self.config_manager.config['robot_arms']['enabled'] = new_state
        print(f"✓ Robot arms {'enabled' if new_state else 'disabled'}")
    
    def test_robot_connections(self):
        """Test robot arm connections"""
        config = self.config_manager.config['robot_arms']
        
        for arm_type in ['follower', 'leader']:
            port = config.get(arm_type, {}).get('port')
            if port:
                print(f"Testing {arm_type} arm connection on {port}...")
                if test_port_connection(port, 1000000):
                    print(f"✓ {arm_type.title()} arm connection successful!")
                else:
                    print(f"✗ {arm_type.title()} arm connection failed.")
            else:
                print(f"No port configured for {arm_type} arm.")

class CameraSetup:
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def check_video_devices_info(self):
        """Check detailed information about video devices"""
        print("\n🔍 Video Device Information:")
        print("=" * 50)
        
        # Check /dev/video* devices
        video_devices = sorted(glob.glob('/dev/video*'))
        
        if not video_devices:
            print("No /dev/video* devices found")
            return
        
        for device in video_devices:
            try:
                # Try to get device info using v4l2-ctl if available
                import subprocess
                try:
                    result = subprocess.run(['v4l2-ctl', '--device', device, '--info'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"\n{device}:")
                        print(result.stdout)
                    else:
                        print(f"{device}: Unable to get detailed info")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    print(f"{device}: v4l2-ctl not available or timeout")
                
                # Basic check - try to open with OpenCV
                cam_id = int(device.replace('/dev/video', ''))
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"{device} (ID {cam_id}): ✓ Working - {w}x{h}")
                    else:
                        print(f"{device} (ID {cam_id}): ✗ Opens but no frame")
                    cap.release()
                else:
                    print(f"{device} (ID {cam_id}): ✗ Cannot open with OpenCV")
                    
            except Exception as e:
                print(f"{device}: Error - {e}")
    
    def show_current_config(self):
        """Display current camera configuration"""
        config = self.config_manager.config.get('cameras', {})
        print("\nCurrent Camera Configuration:")
        print("=" * 40)
        print(f"Enabled:     {config.get('enabled', True)}")
        print(f"Auto-detect: {config.get('auto_detect', True)}")
        print("\nCamera 1:")
        print(f"  ID:        {config.get('camera1', {}).get('id', 0)}")
        print(f"  Enabled:   {config.get('camera1', {}).get('enabled', True)}")
        print("\nCamera 2:")
        print(f"  ID:        {config.get('camera2', {}).get('id', 2)}")
        print(f"  Enabled:   {config.get('camera2', {}).get('enabled', True)}")
    
    def get_available_video_devices(self):
        """Get available video devices from /dev/video* - very permissive"""
        video_devices = sorted(glob.glob('/dev/video*'))
        camera_ids = set()
        
        print(f"Debug: Found /dev/video* devices: {video_devices}")
        
        for device in video_devices:
            try:
                cam_id = int(device.replace('/dev/video', ''))
                # Test ALL devices - don't filter metadata devices
                camera_ids.add(cam_id)
                print(f"Debug: Added camera ID {cam_id} from {device}")
            except ValueError:
                continue
        
        # If no /dev/video* devices found, test a very wide range
        if not camera_ids:
            print("Debug: No /dev/video* devices found, testing range 0-20")
            camera_ids = set(range(21))
        else:
            # Also add some common IDs that might not have /dev/video* files
            common_ids = [0, 1, 2, 3, 4, 5]
            for cid in common_ids:
                if cid not in camera_ids:
                    camera_ids.add(cid)
                    print(f"Debug: Added common camera ID {cid}")
        
        return sorted(camera_ids)
    
    def auto_detect_cameras(self):
        """Auto-detect available cameras"""
        print("\n📷 Scanning for cameras...")
        
        # Check /dev/video* devices first
        video_devices = sorted(glob.glob('/dev/video*'))
        if video_devices:
            print(f"Found video devices: {video_devices}")
        
        camera_ids = self.get_available_video_devices()
        available_cameras = []
        
        print(f"Testing camera IDs: {camera_ids}")
        
        for cam_id in camera_ids:
            print(f"Testing camera {cam_id}...", end=" ")
            try:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    # Set basic camera properties
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Give minimal time for camera to initialize
                    time.sleep(0.2)
                    
                    # Try to read at least one frame - very lenient check
                    success = False
                    for attempt in range(3):  # Only 3 quick attempts
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            h, w = frame.shape[:2]
                            # Very basic check - any reasonable size is OK
                            if h > 0 and w > 0 and h < 10000 and w < 10000:
                                available_cameras.append({'id': cam_id, 'resolution': f"{w}x{h}"})
                                print(f"✓ {w}x{h}")
                                success = True
                                break
                        time.sleep(0.1)
                    
                    if not success:
                        # Even if no frame, but camera opened - still count as available
                        available_cameras.append({'id': cam_id, 'resolution': 'Unknown'})
                        print("✓ Opens (no frame)")
                    
                    cap.release()
                else:
                    print("✗ Failed to open")
            except Exception as e:
                print(f"✗ Error: {e}")
            
            # Minimal delay
            time.sleep(0.1)
        
        print(f"\nFound {len(available_cameras)} working cameras:")
        for cam in available_cameras:
            print(f"  Camera {cam['id']}: {cam['resolution']}")
        
        if len(available_cameras) >= 2:
            self.config_manager.config['cameras']['camera1']['id'] = available_cameras[0]['id']
            self.config_manager.config['cameras']['camera2']['id'] = available_cameras[1]['id']
            self.config_manager.config['cameras']['camera2']['enabled'] = True
            print(f"✓ Auto-assigned Camera 1: ID {available_cameras[0]['id']}")
            print(f"✓ Auto-assigned Camera 2: ID {available_cameras[1]['id']}")
        elif len(available_cameras) == 1:
            self.config_manager.config['cameras']['camera1']['id'] = available_cameras[0]['id']
            self.config_manager.config['cameras']['camera2']['enabled'] = False
            print(f"✓ Auto-assigned Camera 1: ID {available_cameras[0]['id']}")
            print("⚠ Only one camera found, Camera 2 disabled")
        else:
            print("No cameras detected.")
            print("Try manually checking with: ls /dev/video*")
        
        return len(available_cameras) > 0
    
    def test_single_camera(self, cam_id):
        """Test a single camera and capture sample image - very lenient"""
        try:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                return None, "Failed to open camera"
            
            # Set basic properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Minimal wait time
            time.sleep(0.2)
            
            # Try to get any frame - very lenient
            best_frame = None
            resolution = "Unknown"
            
            for i in range(5):  # Only 5 attempts
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    if h > 0 and w > 0:  # Any size is acceptable
                        best_frame = frame.copy()
                        resolution = f"{w}x{h}"
                        break
                time.sleep(0.1)
            
            cap.release()
            
            # Save sample image if we got any frame
            sample_filename = f"camera_{cam_id}_test.jpg"
            if best_frame is not None:
                cv2.imwrite(sample_filename, best_frame)
                return {
                    'id': cam_id,
                    'resolution': resolution,
                    'sample_file': sample_filename
                }, None
            else:
                # Even without a frame, if camera opened, create a placeholder
                return {
                    'id': cam_id,
                    'resolution': 'Unknown',
                    'sample_file': None
                }, None
            
        except Exception as e:
            return None, str(e)
    
    def preview_cameras(self):
        """Preview cameras with sample images"""
        print("\n📸 Camera Preview Mode")
        print("=" * 40)
        
        camera_ids = self.get_available_video_devices()
        working_cameras = []
        
        print("Testing cameras and capturing samples...")
        for cam_id in camera_ids:
            print(f"Testing Camera {cam_id}...", end=" ")
            camera_info, error = self.test_single_camera(cam_id)
            
            if camera_info:
                working_cameras.append(camera_info)
                sample_info = f"Sample: {camera_info['sample_file']}" if camera_info['sample_file'] else "No sample"
                print(f"✓ {camera_info['resolution']} - {sample_info}")
            else:
                print(f"✗ {error}")
        
        if not working_cameras:
            print("No working cameras found.")
            return
        
        print(f"\nFound {len(working_cameras)} working cameras:")
        for i, cam in enumerate(working_cameras, 1):
            print(f"{i}. Camera ID {cam['id']}: {cam['resolution']}")
        
        print("\nSample images saved. You can view them to see camera output.")
        
        # Auto-assign cameras based on sample images created
        print(f"\nAuto-assigning cameras based on {len(working_cameras)} working cameras...")
        if len(working_cameras) >= 1:
            cam_id = working_cameras[0]['id']
            self.config_manager.config['cameras']['camera1']['id'] = cam_id
            self.config_manager.config['cameras']['camera1']['enabled'] = True
            print(f"✓ Auto-assigned Camera 1: ID {cam_id}")
        
        if len(working_cameras) >= 2:
            cam_id = working_cameras[1]['id']
            self.config_manager.config['cameras']['camera2']['id'] = cam_id
            self.config_manager.config['cameras']['camera2']['enabled'] = True
            print(f"✓ Auto-assigned Camera 2: ID {cam_id}")
        elif len(working_cameras) == 1:
            self.config_manager.config['cameras']['camera2']['enabled'] = False
            print("Camera 2 disabled (only 1 camera found)")
        
        # Clean up sample images
        cleanup = input("\nDelete sample images? (Y/n): ").strip().lower()
        if not cleanup.startswith('n'):
            for cam in working_cameras:
                try:
                    if cam['sample_file']:
                        os.remove(cam['sample_file'])
                except:
                    pass
    
    def configure_cameras(self):
        """Interactive camera configuration"""
        print("\n" + "=" * 50)
        print("📷 CAMERA SETUP")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Show current configuration")
            print("2. Auto-detect cameras")
            print("3. Preview cameras with samples")
            print("4. Check video device info")
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
                self.preview_cameras()
            elif choice == '4':
                self.check_video_devices_info()
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
        camera_ids = self.get_available_video_devices()
        print(f"Available camera IDs: {camera_ids}")
        
        try:
            cam_id = int(input(f"Enter camera ID for {camera_name}: ").strip())
            if cam_id in camera_ids:
                self.config_manager.config['cameras'][camera_name]['id'] = cam_id
                print(f"✓ {camera_name.title()} ID set to: {cam_id}")
            else:
                print(f"Camera ID {cam_id} not available.")
        except ValueError:
            print("Invalid camera ID.")
    
    def toggle_enable(self):
        """Toggle cameras enable/disable"""
        current = self.config_manager.config['cameras'].get('enabled', True)
        new_state = not current
        self.config_manager.config['cameras']['enabled'] = new_state
        print(f"✓ Cameras {'enabled' if new_state else 'disabled'}")
    
    def test_cameras(self):
        """Test camera connections"""
        config = self.config_manager.config['cameras']
        
        for camera_name in ['camera1', 'camera2']:
            cam_config = config.get(camera_name, {})
            if cam_config.get('enabled', True):
                cam_id = cam_config.get('id', 0)
                print(f"Testing {camera_name} (ID: {cam_id})...")
                
                camera_info, error = self.test_single_camera(cam_id)
                if camera_info:
                    print(f"✓ {camera_name.title()} working: {camera_info['resolution']}")
                    # Clean up test image
                    try:
                        os.remove(camera_info['sample_file'])
                    except:
                        pass
                else:
                    print(f"✗ {camera_name.title()} failed: {error}")

def main():
    """Main hardware setup tool"""
    print("🤖 Robot Control System - Hardware Setup Tool")
    print("=" * 60)
    
    config_manager = HardwareConfigManager()
    arduino_setup = ArduinoSetup(config_manager)
    robot_arms_setup = RobotArmsSetup(config_manager)
    camera_setup = CameraSetup(config_manager)
    
    while True:
        print("\n🔧 Hardware Components:")
        print("1. 🔌 Arduino Controller Setup")
        print("2. 🦾 Robot Arms Setup") 
        print("3. 📷 Camera Setup")
        print("4. 🔍 Auto-detect All Hardware")
        print("5. 📋 Show All Configurations")
        print("6. 💾 Save Configuration")
        print("7. ⚙️  Update Main Config")
        print("8. 🧪 Run Connection Test")
        print("0. 🚪 Exit")
        
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
                print("\n🔍 Auto-detecting all hardware...")
                arduino_setup.auto_detect_arduino()
                robot_arms_setup.auto_detect_robot_arms()
                camera_setup.auto_detect_cameras()
                print("✓ Auto-detection complete!")
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
    
    print("\n👋 Hardware setup tool closed.")

if __name__ == "__main__":
    main()