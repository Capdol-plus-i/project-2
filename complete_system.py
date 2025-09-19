#!/usr/bin/env python3
"""
Complete Integrated System
- Hand tracking from dual cameras
- Leader-follower robot arm synchronization
- Data logging
All in one script
"""

import os
import sys
import time
import json
import csv
import threading
import signal
from datetime import datetime
import argparse
import cv2

# Set TensorFlow logging level
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Robot arm imports
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

class CompleteSystem:
    def __init__(self):
        self.recording = False
        self.snapshot_mode = False
        self.sync_active = False
        self.running = True
        self.lock = threading.Lock()

        # Camera settings
        self.WIDTH = int(os.environ.get("W", 640))
        self.HEIGHT = int(os.environ.get("H", 480))
        self.FPS = int(os.environ.get("FPS", 30))

        # Camera objects
        self.cap1 = None
        self.cap2 = None
        self.hands1 = None
        self.hands2 = None
        self.use_mp = False

        # Robot arm settings
        self.load_hardware_config()
        self.init_robot_config()

        # Robot arm objects
        self.leader_port_handler = None
        self.leader_packet_handler = None
        self.follower_port_handler = None
        self.follower_packet_handler = None
        self.connected_leader_motors = []
        self.connected_follower_motors = []

        # Output file
        self.output_file = None
        self.csv_writer = None

        # Synchronization settings (will be loaded from calibration.json)
        self.position_offsets = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.direction_multipliers = {1: 1, 2: 1, 3: 1, 4: 1}
        self.id_map = {1: 1, 2: 2, 3: 3, 4: 4}

        # Load calibration if available
        self.load_calibration()

        # Sync thread
        self.sync_thread = None

    def load_calibration(self, filename="calibration.json"):
        """Load calibration offsets from file"""
        try:
            if not os.path.exists(filename):
                print(f"ℹ️ No calibration file found ({filename}), using defaults")
                return False

            with open(filename, 'r') as f:
                calibration_data = json.load(f)

            # Load offsets
            if 'position_offsets' in calibration_data:
                offsets = calibration_data['position_offsets']
                for key, value in offsets.items():
                    motor_id = int(key)
                    self.position_offsets[motor_id] = float(value)

            # Load direction multipliers
            if 'direction_multipliers' in calibration_data:
                multipliers = calibration_data['direction_multipliers']
                for key, value in multipliers.items():
                    motor_id = int(key)
                    self.direction_multipliers[motor_id] = int(value)

            # Load ID mapping
            if 'id_map' in calibration_data:
                id_mapping = calibration_data['id_map']
                for key, value in id_mapping.items():
                    leader_id = int(key)
                    follower_id = int(value)
                    self.id_map[leader_id] = follower_id

            print(f"✅ Calibration loaded from {filename}")
            print("📋 Calibration settings:")
            for leader_id in self.MOTOR_IDS:
                follower_id = self.id_map.get(leader_id, leader_id)
                offset = self.position_offsets.get(follower_id, 0.0)
                direction = self.direction_multipliers.get(follower_id, 1)
                print(f"  L{leader_id}→F{follower_id}: offset {offset:+6.2f}° | dir {direction:+d}")

            return True

        except Exception as e:
            print(f"⚠️ Failed to load calibration: {e}")
            print("Using default calibration (no offsets)")
            return False

    def load_hardware_config(self):
        """Load hardware configuration from hardware_config.json"""
        config_file = 'hardware_config.json'
        default_config = {
            'leader': {'port': '/dev/leader_arm', 'baudrate': 1000000},
            'follower': {'port': '/dev/follower_arm', 'baudrate': 1000000}
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    robot_config = config.get('robot_arms', {})
                    self.hw_config = {
                        'leader': {
                            'port': robot_config.get('leader', {}).get('port', default_config['leader']['port']),
                            'baudrate': robot_config.get('leader', {}).get('baudrate', default_config['leader']['baudrate'])
                        },
                        'follower': {
                            'port': robot_config.get('follower', {}).get('port', default_config['follower']['port']),
                            'baudrate': robot_config.get('follower', {}).get('baudrate', default_config['follower']['baudrate'])
                        }
                    }
            except Exception as e:
                print(f"Warning: Failed to load hardware config: {e}")
                self.hw_config = default_config
        else:
            self.hw_config = default_config

    def init_robot_config(self):
        """Initialize robot arm configuration"""
        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.MOTOR_IDS = [1, 2, 3, 4]

        self.LEADER_CONFIG = {
            'port': self.hw_config['leader']['port'],
            'baudrate': self.hw_config['leader']['baudrate'],
            'motors': {
                1: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088},
                2: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088},
                3: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088},
                4: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088}
            }
        }

        self.FOLLOWER_CONFIG = {
            'port': self.hw_config['follower']['port'],
            'baudrate': self.hw_config['follower']['baudrate'],
            'motors': {
                1: {'model': 'XL430-W250-T', 'center': 2048, 'resolution': 0.088},
                2: {'model': 'XL430-W250-T', 'center': 2048, 'resolution': 0.088},
                3: {'model': 'XL430-W250-T', 'center': 2048, 'resolution': 0.088},
                4: {'model': 'XL330-M288-T', 'center': 2048, 'resolution': 0.088}
            }
        }

    def init_cameras(self):
        """Initialize dual cameras"""
        print("🎥 Initializing cameras...")

        # Initialize first camera
        self.cap1 = self.open_camera(0)
        if not self.cap1:
            print("❌ Failed to open first camera")
            return False
        print("✓ First camera initialized")

        # Initialize second camera
        self.cap2 = self.open_camera(2)
        if self.cap2:
            print("✓ Second camera initialized")
        else:
            print("⚠️ Second camera not available")

        # Initialize MediaPipe
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils

            self.hands1 = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )

            if self.cap2:
                self.hands2 = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5
                )

            self.use_mp = True
            print("✓ MediaPipe initialized")
        except Exception as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            self.use_mp = False

        return True

    def open_camera(self, device_id):
        """Open camera with optimized settings"""
        if isinstance(device_id, str) and device_id.startswith('/dev/'):
            try:
                device_id = int(device_id.split('video')[1])
            except:
                device_id = 0

        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self.FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    def init_robot_arms(self):
        """Initialize both robot arms"""
        print("🤖 Initializing robot arms...")

        # Initialize leader arm
        self.leader_port_handler = PortHandler(self.LEADER_CONFIG['port'])
        self.leader_packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.leader_port_handler.openPort():
            print(f"❌ Failed to open leader port {self.LEADER_CONFIG['port']}")
            return False

        if not self.leader_port_handler.setBaudRate(self.LEADER_CONFIG['baudrate']):
            print(f"❌ Failed to set leader baudrate")
            return False

        print(f"✓ Leader arm connected: {self.LEADER_CONFIG['port']}")

        # Initialize follower arm
        self.follower_port_handler = PortHandler(self.FOLLOWER_CONFIG['port'])
        self.follower_packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.follower_port_handler.openPort():
            print(f"❌ Failed to open follower port {self.FOLLOWER_CONFIG['port']}")
            return False

        if not self.follower_port_handler.setBaudRate(self.FOLLOWER_CONFIG['baudrate']):
            print(f"❌ Failed to set follower baudrate")
            return False

        print(f"✓ Follower arm connected: {self.FOLLOWER_CONFIG['port']}")

        # Ping motors
        self.connected_leader_motors = self.ping_motors(
            self.leader_port_handler, self.leader_packet_handler, "Leader")
        self.connected_follower_motors = self.ping_motors(
            self.follower_port_handler, self.follower_packet_handler, "Follower")

        print(f"✓ Leader motors: {self.connected_leader_motors}")
        print(f"✓ Follower motors: {self.connected_follower_motors}")

        return len(self.connected_leader_motors) > 0 and len(self.connected_follower_motors) > 0

    def ping_motors(self, port_handler, packet_handler, name):
        """Ping motors and return connected ones"""
        connected_motors = []
        for motor_id in self.MOTOR_IDS:
            try:
                model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
                if comm_result == COMM_SUCCESS:
                    connected_motors.append(motor_id)
                    print(f"  ✓ {name} Motor {motor_id}: Connected")
            except Exception as e:
                print(f"  ❌ {name} Motor {motor_id}: Failed ({e})")
        return connected_motors

    def read_motor_position(self, port_handler, packet_handler, motor_id, config):
        """Read current position from a specific motor"""
        try:
            position, comm_result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, self.ADDR_PRESENT_POSITION)

            if comm_result == COMM_SUCCESS and error == 0:
                motor_config = config['motors'][motor_id]
                angle = (position - motor_config['center']) * motor_config['resolution']
                return position, angle, True
            else:
                return None, None, False
        except:
            return None, None, False

    def set_motor_position(self, port_handler, packet_handler, motor_id, position):
        """Set goal position for a specific motor"""
        try:
            position = max(0, min(4095, int(position)))
            comm_result, error = packet_handler.write4ByteTxRx(
                port_handler, motor_id, self.ADDR_GOAL_POSITION, position)
            return comm_result == COMM_SUCCESS and error == 0
        except:
            return False

    def set_torque_state(self, port_handler, packet_handler, motor_id, enable):
        """Set torque state for a specific motor"""
        try:
            comm_result, error = packet_handler.write1ByteTxRx(
                port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 1 if enable else 0)
            return comm_result == COMM_SUCCESS and error == 0
        except:
            return False

    def enable_follower_torques(self):
        """Enable torque for all connected follower motors"""
        print("⚡ Enabling follower arm torques...")

        success_count = 0
        for motor_id in self.connected_follower_motors:
            if self.set_torque_state(self.follower_port_handler, self.follower_packet_handler, motor_id, True):
                print(f"  ✓ Follower Motor {motor_id} torque enabled")
                success_count += 1
            else:
                print(f"  ✗ Follower Motor {motor_id} torque enable failed")

        return success_count == len(self.connected_follower_motors)

    def disable_follower_torques(self):
        """Disable torque for all connected follower motors"""
        print("⚡ Disabling follower arm torques...")

        for motor_id in self.connected_follower_motors:
            if self.set_torque_state(self.follower_port_handler, self.follower_packet_handler, motor_id, False):
                print(f"  ✓ Follower Motor {motor_id} torque disabled")
            else:
                print(f"  ✗ Follower Motor {motor_id} torque disable failed")

    def angle_to_position(self, angle, motor_id, config):
        """Convert angle to position for specific motor"""
        motor_config = config['motors'][motor_id]
        return int(motor_config['center'] + (angle / motor_config['resolution']))

    def sync_leader_to_follower(self):
        """Real-time synchronization thread - copies leader positions to follower"""
        print("🔄 Leader-follower synchronization started")

        last_positions = {}
        update_count = 0
        error_count = 0
        max_errors = 10

        while self.sync_active and self.running:
            try:
                positions_changed = False
                current_leader_positions = {}

                # Read all leader motor positions
                for leader_id in self.connected_leader_motors:
                    follower_id = self.id_map.get(leader_id)
                    if follower_id not in self.connected_follower_motors:
                        continue

                    position, angle, success = self.read_motor_position(
                        self.leader_port_handler, self.leader_packet_handler, leader_id, self.LEADER_CONFIG)

                    if success:
                        current_leader_positions[leader_id] = {'position': position, 'angle': angle}

                        # Check if position changed significantly
                        if (leader_id not in last_positions or
                            abs(position - last_positions[leader_id]['position']) > 2):
                            positions_changed = True
                    else:
                        error_count += 1
                        if error_count > max_errors:
                            print(f"\n⚠️ Too many read errors ({error_count}), stopping sync")
                            self.sync_active = False
                            break
                        continue

                # If leader positions changed, update follower positions
                if positions_changed and current_leader_positions:
                    success_count = 0

                    for leader_id, leader_data in current_leader_positions.items():
                        follower_id = self.id_map.get(leader_id)
                        if follower_id in self.connected_follower_motors:
                            # Apply calibration offset & mapping
                            d = self.direction_multipliers.get(follower_id, 1)
                            offset = self.position_offsets.get(follower_id, 0.0)
                            target_angle = d * leader_data['angle'] + offset
                            target_position = self.angle_to_position(target_angle, follower_id, self.FOLLOWER_CONFIG)

                            # Set follower motor position
                            if self.set_motor_position(self.follower_port_handler, self.follower_packet_handler,
                                                    follower_id, target_position):
                                success_count += 1

                    # Reset error count on successful sync
                    error_count = 0

                last_positions = current_leader_positions.copy()
                update_count += 1

                # Sleep for smooth operation (20Hz)
                time.sleep(0.05)

            except Exception as e:
                print(f"\n❌ Sync error: {e}")
                error_count += 1
                if error_count > max_errors:
                    self.sync_active = False
                    break
                time.sleep(0.1)

        print("\n🛑 Leader-follower synchronization stopped")

    def start_synchronization(self):
        """Start leader-follower synchronization"""
        if self.sync_active:
            print("⚠️ Synchronization is already running")
            return False

        # Enable follower torques
        if not self.enable_follower_torques():
            print("❌ Failed to enable follower torques")
            return False

        # Start sync thread
        self.sync_active = True
        self.sync_thread = threading.Thread(target=self.sync_leader_to_follower, daemon=True)
        self.sync_thread.start()

        print("✅ Leader-follower synchronization started")
        return True

    def stop_synchronization(self):
        """Stop leader-follower synchronization"""
        if not self.sync_active:
            return

        self.sync_active = False
        print("⏳ Stopping synchronization...")
        time.sleep(0.2)

        # Disable follower torques
        self.disable_follower_torques()

    def get_hand_coordinates(self):
        """Get hand coordinates from both cameras"""
        coords = {
            'cam1_x': None, 'cam1_y': None,
            'cam2_x': None, 'cam2_y': None
        }

        # Process camera 1
        if self.cap1:
            frame1, finger_coords1 = self.process_frame(self.cap1, self.hands1)
            if finger_coords1:
                coords['cam1_x'] = finger_coords1[0]['index_tip'][0]
                coords['cam1_y'] = finger_coords1[0]['index_tip'][1]

        # Process camera 2
        if self.cap2:
            frame2, finger_coords2 = self.process_frame(self.cap2, self.hands2)
            if finger_coords2:
                coords['cam2_x'] = finger_coords2[0]['index_tip'][0]
                coords['cam2_y'] = finger_coords2[0]['index_tip'][1]

        return coords

    def process_frame(self, cap, hands_instance):
        """Process single camera frame"""
        ok, frame = cap.read()
        if not ok:
            return None, []

        finger_coords = []
        if self.use_mp and hands_instance:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_instance.process(rgb)
            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    h, w, _ = frame.shape
                    idx_x, idx_y = int(index_tip.x * w), int(index_tip.y * h)

                    finger_coords.append({
                        'index_tip': (idx_x, idx_y, index_tip.x, index_tip.y)
                    })

        return frame, finger_coords

    def get_robot_positions(self):
        """Get current positions from follower robot arm"""
        positions = {
            'follower_pos1': None, 'follower_pos2': None,
            'follower_pos3': None, 'follower_pos4': None
        }

        for i, motor_id in enumerate([1, 2, 3, 4], 1):
            if motor_id in self.connected_follower_motors:
                position, angle, success = self.read_motor_position(
                    self.follower_port_handler, self.follower_packet_handler, motor_id, self.FOLLOWER_CONFIG)
                if success:
                    positions[f'follower_pos{i}'] = position

        return positions

    def init_output_file(self, filename=None):
        """Initialize CSV output file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complete_log_{timestamp}.csv"

        self.output_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.output_file)

        # Write header
        header = [
            'timestamp', 'cam1_x', 'cam1_y', 'cam2_x', 'cam2_y',
            'follower_pos1', 'follower_pos2', 'follower_pos3', 'follower_pos4'
        ]
        self.csv_writer.writerow(header)
        self.output_file.flush()

        print(f"📝 Output file: {filename}")
        return filename

    def collect_data_point(self):
        """Collect single data point from all sources"""
        timestamp = time.time()

        # Get hand coordinates
        hand_coords = self.get_hand_coordinates()

        # Get robot positions
        robot_positions = self.get_robot_positions()

        # Combine data
        data_point = {
            'timestamp': timestamp,
            **hand_coords,
            **robot_positions
        }

        return data_point

    def record_data_point(self, data_point):
        """Record data point to CSV file"""
        if self.csv_writer:
            row = [
                data_point['timestamp'],
                data_point['cam1_x'], data_point['cam1_y'],
                data_point['cam2_x'], data_point['cam2_y'],
                data_point['follower_pos1'], data_point['follower_pos2'],
                data_point['follower_pos3'], data_point['follower_pos4']
            ]
            self.csv_writer.writerow(row)
            self.output_file.flush()

    def start_continuous_recording(self):
        """Start continuous data recording"""
        print("🔴 Starting continuous recording...")
        print("📍 Press Ctrl+C to stop recording")

        self.recording = True
        count = 0

        try:
            while self.recording and self.running:
                data_point = self.collect_data_point()
                self.record_data_point(data_point)
                count += 1

                if count % 10 == 0:
                    hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                    robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"
                    sync_status = "🔄" if self.sync_active else "⏸️"

                    print(f"\r📊 [{count:4d}] {hand_status} {robot_status} {sync_status} | "
                          f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                          f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                          f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                          f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="", flush=True)

                time.sleep(0.1)

        except KeyboardInterrupt:
            self.recording = False

        print(f"\n🛑 Recording stopped. Total: {count} data points")

    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")

        self.recording = False
        self.snapshot_mode = False
        self.running = False

        # Stop synchronization
        if self.sync_active:
            self.stop_synchronization()
            time.sleep(0.5)

        # Close cameras
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        cv2.destroyAllWindows()

        # Close robot arm connections
        if self.leader_port_handler:
            self.leader_port_handler.closePort()
        if self.follower_port_handler:
            self.follower_port_handler.closePort()

        # Close output file
        if self.output_file:
            self.output_file.close()

        print("✅ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="Complete Integrated System")
    parser.add_argument("--mode", choices=['continuous', 'test'], default='continuous',
                       help="Mode: continuous recording or test")
    parser.add_argument("--output", help="Output CSV file name")
    parser.add_argument("--no-sync", action='store_true', help="Don't start leader-follower sync automatically")
    args = parser.parse_args()

    system = CompleteSystem()

    # Setup signal handler
    def signal_handler(sig, frame):
        system.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize all systems
        print("🚀 Initializing Complete System...")
        print("=" * 60)

        if not system.init_cameras():
            print("⚠️ Camera initialization failed, continuing with robot only")

        if not system.init_robot_arms():
            print("❌ Robot arm initialization failed")
            return

        # Start leader-follower synchronization
        if not args.no_sync:
            if system.start_synchronization():
                print("✅ Leader-follower synchronization active")
            else:
                print("⚠️ Failed to start synchronization")

        if args.mode == 'test':
            print("🧪 Test mode - showing live data")
            print("Press Ctrl+C to stop")

            while True:
                data_point = system.collect_data_point()
                hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"
                sync_status = "🔄" if system.sync_active else "⏸️"

                print(f"\r{hand_status} {robot_status} {sync_status} | "
                      f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                      f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                      f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                      f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="", flush=True)
                time.sleep(0.2)
        else:
            # Initialize output file
            output_filename = system.init_output_file(args.output)

            print("✅ All systems initialized successfully")
            print("🔄 Leader-follower sync is active - move leader arm to see follower follow")
            print(f"📊 Data format: timestamp, cam1_x, cam1_y, cam2_x, cam2_y, follower_pos1-4")
            print("=" * 60)

            # Start continuous recording
            system.start_continuous_recording()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()