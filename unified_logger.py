#!/usr/bin/env python3
"""
Unified Hand Tracking and Robot Arm Logger
All-in-one solution without port conflicts
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

class UnifiedLogger:
    def __init__(self):
        self.recording = False
        self.snapshot_mode = False
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
        self.connected_leader_motors = []
        self.follower_port_handler = None
        self.follower_packet_handler = None
        self.connected_follower_motors = []

        # Output file
        self.output_file = None
        self.csv_writer = None

        # Last known robot positions
        self.last_robot_positions = {
            'follower_pos1': None, 'follower_pos2': None,
            'follower_pos3': None, 'follower_pos4': None
        }

        # Sync control
        self.sync_active = False
        self.sync_thread = None

        # Calibration data
        self.position_offsets = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.direction_multipliers = {1: 1, 2: 1, 3: 1, 4: 1}
        self.id_map = {1: 1, 2: 2, 3: 3, 4: 4}

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
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_TORQUE_ENABLE = 64
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

    def load_calibration(self, filename="calibration.json"):
        """Load calibration offsets from file"""
        try:
            if not os.path.exists(filename):
                print(f"❌ Calibration file {filename} not found")
                return False

            with open(filename, 'r') as f:
                calibration_data = json.load(f)

            self.position_offsets = calibration_data.get('position_offsets', {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
            self.direction_multipliers = calibration_data.get('direction_multipliers', {1: 1, 2: 1, 3: 1, 4: 1})
            self.id_map = calibration_data.get('id_map', {1: 1, 2: 2, 3: 3, 4: 4})

            # Convert string keys to int if needed
            if isinstance(list(self.position_offsets.keys())[0], str):
                self.position_offsets = {int(k): v for k, v in self.position_offsets.items()}
            if isinstance(list(self.direction_multipliers.keys())[0], str):
                self.direction_multipliers = {int(k): int(v) for k, v in self.direction_multipliers.items()}
            if isinstance(list(self.id_map.keys())[0], str):
                self.id_map = {int(k): int(v) for k, v in self.id_map.items()}

            print(f"📁 Calibration loaded from {filename}")
            for leader_id in self.MOTOR_IDS:
                follower_id = self.id_map.get(leader_id)
                off = self.position_offsets.get(follower_id, 0.0)
                d = self.direction_multipliers.get(follower_id, 1)
                print(f"  L{leader_id}→F{follower_id}: offset {off:+6.2f}° | dir {d:+d}")

            return True
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False

    def init_leader_arm(self):
        """Initialize leader robot arm connection"""
        print("🎯 Initializing leader robot arm...")

        self.leader_port_handler = PortHandler(self.LEADER_CONFIG['port'])
        self.leader_packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.leader_port_handler.openPort():
            print(f"❌ Failed to open leader port {self.LEADER_CONFIG['port']}")
            return False

        if not self.leader_port_handler.setBaudRate(self.LEADER_CONFIG['baudrate']):
            print(f"❌ Failed to set leader baudrate")
            return False

        print(f"✓ Leader arm connected: {self.LEADER_CONFIG['port']}")

        # Ping leader motors
        self.connected_leader_motors = []
        for motor_id in self.MOTOR_IDS:
            try:
                model_number, comm_result, error = self.leader_packet_handler.ping(
                    self.leader_port_handler, motor_id)
                if comm_result == COMM_SUCCESS:
                    self.connected_leader_motors.append(motor_id)
                    print(f"  ✓ Leader Motor {motor_id}: Connected")
            except Exception as e:
                print(f"  ❌ Leader Motor {motor_id}: Failed ({e})")

        print(f"✓ Leader motors: {self.connected_leader_motors}")
        return len(self.connected_leader_motors) > 0

    def init_robot_arm(self):
        """Initialize follower robot arm connection"""
        print("🤖 Initializing follower robot arm...")

        self.follower_port_handler = PortHandler(self.FOLLOWER_CONFIG['port'])
        self.follower_packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.follower_port_handler.openPort():
            print(f"❌ Failed to open follower port {self.FOLLOWER_CONFIG['port']}")
            return False

        if not self.follower_port_handler.setBaudRate(self.FOLLOWER_CONFIG['baudrate']):
            print(f"❌ Failed to set follower baudrate")
            return False

        print(f"✓ Follower arm connected: {self.FOLLOWER_CONFIG['port']}")

        # Ping follower motors
        self.connected_follower_motors = []
        for motor_id in self.MOTOR_IDS:
            try:
                model_number, comm_result, error = self.follower_packet_handler.ping(
                    self.follower_port_handler, motor_id)
                if comm_result == COMM_SUCCESS:
                    self.connected_follower_motors.append(motor_id)
                    print(f"  ✓ Motor {motor_id}: Connected")
            except Exception as e:
                print(f"  ❌ Motor {motor_id}: Failed ({e})")

        print(f"✓ Follower motors: {self.connected_follower_motors}")
        return len(self.connected_follower_motors) > 0

    def get_hand_coordinates_with_frames(self, draw_landmarks=False):
        """Get hand coordinates from both cameras with frames"""
        coords = {
            'cam1_x': None, 'cam1_y': None,
            'cam2_x': None, 'cam2_y': None
        }
        frames = {'frame1': None, 'frame2': None}

        # Process camera 1
        if self.cap1:
            frame1, finger_coords1 = self.process_frame(self.cap1, self.hands1, draw_landmarks)
            frames['frame1'] = frame1
            if finger_coords1:
                coords['cam1_x'] = finger_coords1[0]['index_tip'][0]
                coords['cam1_y'] = finger_coords1[0]['index_tip'][1]

        # Process camera 2
        if self.cap2:
            frame2, finger_coords2 = self.process_frame(self.cap2, self.hands2, draw_landmarks)
            frames['frame2'] = frame2
            if finger_coords2:
                coords['cam2_x'] = finger_coords2[0]['index_tip'][0]
                coords['cam2_y'] = finger_coords2[0]['index_tip'][1]

        return coords, frames

    def get_hand_coordinates(self):
        """Get hand coordinates from both cameras"""
        coords, _ = self.get_hand_coordinates_with_frames()
        return coords

    def process_frame(self, cap, hands_instance, draw_landmarks=False):
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
                    # Draw hand landmarks if requested
                    if draw_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    index_tip = hand_landmarks.landmark[8]
                    h, w, _ = frame.shape
                    idx_x, idx_y = int(index_tip.x * w), int(index_tip.y * h)

                    # Draw index finger tip
                    if draw_landmarks:
                        cv2.circle(frame, (idx_x, idx_y), 8, (0, 255, 0), -1)
                        cv2.putText(frame, f"({idx_x},{idx_y})", (idx_x + 10, idx_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                try:
                    position, comm_result, error = self.follower_packet_handler.read4ByteTxRx(
                        self.follower_port_handler, motor_id, self.ADDR_PRESENT_POSITION)

                    if comm_result == COMM_SUCCESS and error == 0:
                        positions[f'follower_pos{i}'] = position
                        # Update last known position
                        self.last_robot_positions[f'follower_pos{i}'] = position
                    else:
                        # Use last known position if read fails
                        positions[f'follower_pos{i}'] = self.last_robot_positions[f'follower_pos{i}']
                except Exception as e:
                    # Use last known position on exception
                    positions[f'follower_pos{i}'] = self.last_robot_positions[f'follower_pos{i}']

        return positions

    def read_motor_position(self, port_handler, packet_handler, motor_id, config):
        """Read current position from a specific motor"""
        position, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, self.ADDR_PRESENT_POSITION)

        if comm_result == COMM_SUCCESS and error == 0:
            motor_config = config['motors'][motor_id]
            angle = (position - motor_config['center']) * motor_config['resolution']
            return position, angle, True
        else:
            return None, None, False

    def set_motor_position(self, port_handler, packet_handler, motor_id, position, config):
        """Set goal position for a specific motor"""
        position = max(0, min(4095, int(position)))

        comm_result, error = packet_handler.write4ByteTxRx(
            port_handler, motor_id, self.ADDR_GOAL_POSITION, position)

        return comm_result == COMM_SUCCESS and error == 0

    def set_torque_state(self, port_handler, packet_handler, motor_id, enable):
        """Set torque state for a specific motor"""
        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 1 if enable else 0)

        return comm_result == COMM_SUCCESS and error == 0

    def angle_to_position(self, angle, motor_id, config):
        """Convert angle to position for specific motor"""
        motor_config = config['motors'][motor_id]
        return int(motor_config['center'] + (angle / motor_config['resolution']))

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

    def sync_leader_to_follower(self):
        """Real-time synchronization thread - copies leader positions to follower"""
        print("🔄 Real-time synchronization started")
        print("📍 Move the leader arm to see follower arm follow")
        print("🛑 Press 'q' in main menu to stop synchronization")
        print("-" * 80)

        last_positions = {}
        update_count = 0
        error_count = 0
        max_errors = 10

        while self.sync_active:
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

                        # Check if position changed significantly (>2 units)
                        if (leader_id not in last_positions or
                            abs(position - last_positions[leader_id]['position']) > 2):
                            positions_changed = True
                    else:
                        error_count += 1
                        if error_count > max_errors:
                            print(f"\n⚠️  Too many read errors ({error_count}), stopping sync")
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
                                                    follower_id, target_position, self.FOLLOWER_CONFIG):
                                success_count += 1

                    # Display status every 10 updates or when positions change significantly
                    if positions_changed or update_count % 20 == 0:
                        timestamp = time.strftime('%H:%M:%S')
                        print(f"\r⏰ {timestamp} | Syncing {success_count}/{len(current_leader_positions)} motors | ", end="")

                        for leader_id in self.MOTOR_IDS:
                            if leader_id in current_leader_positions:
                                data = current_leader_positions[leader_id]
                                follower_id = self.id_map.get(leader_id)
                                d = self.direction_multipliers.get(follower_id, 1)
                                off = self.position_offsets.get(follower_id, 0.0)
                                target_angle = d * data['angle'] + off
                                print(f"L{leader_id}→F{follower_id}: {data['angle']:+6.1f}°→{target_angle:+6.1f}° | ", end="")
                            else:
                                print(f"L{leader_id}: ---- | ", end="")
                        print("", end="", flush=True)

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

        print("\n🛑 Synchronization stopped")

    def init_output_file(self, filename=None):
        """Initialize CSV output file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unified_log_{timestamp}.csv"

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
        """Start continuous data recording with live camera display"""
        print("🔴 Starting continuous recording...")
        print("📍 Press Ctrl+C to stop recording")
        print("🎥 Live camera display with hand tracking enabled")

        self.recording = True
        count = 0

        try:
            while self.recording and self.running:
                # Get data with frames for display
                hand_coords, frames = self.get_hand_coordinates_with_frames(draw_landmarks=True)
                robot_positions = self.get_robot_positions()

                # Combine data
                data_point = {
                    'timestamp': time.time(),
                    **hand_coords,
                    **robot_positions
                }

                self.record_data_point(data_point)
                count += 1

                # Display camera frames
                if frames['frame1'] is not None:
                    cv2.putText(frames['frame1'], f"Recording - Frame #{count}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frames['frame1'], "Ctrl+C: Stop Recording",
                              (10, frames['frame1'].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow('Camera 1 - Recording', frames['frame1'])

                if frames['frame2'] is not None:
                    cv2.putText(frames['frame2'], f"Recording - Frame #{count}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Camera 2 - Recording', frames['frame2'])

                # Update terminal display every 10 samples
                if count % 10 == 0:
                    hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                    robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"
                    sync_status = "🔄" if self.sync_active else "⏸️"

                    print(f"\r📊 [{count:4d}] {hand_status} {robot_status} {sync_status} | "
                          f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                          f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                          f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                          f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="", flush=True)

                # Check for ESC key to stop (optional)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                time.sleep(0.1)  # 10Hz sampling rate

        except KeyboardInterrupt:
            self.recording = False

        cv2.destroyAllWindows()
        print(f"\n🛑 Recording stopped. Total: {count} data points")

    def start_snapshot_mode(self):
        """Start snapshot mode - record on demand with live camera display"""
        print("📸 Snapshot mode activated")
        print("📍 Press SPACE to take snapshot, ESC to quit")
        print("🎥 Live camera display with hand tracking enabled")

        self.snapshot_mode = True
        snapshot_count = 0

        try:
            while self.snapshot_mode and self.running:
                # Get current data and frames for preview
                hand_coords, frames = self.get_hand_coordinates_with_frames(draw_landmarks=True)
                robot_positions = self.get_robot_positions()

                # Combine data
                data_point = {
                    'timestamp': time.time(),
                    **hand_coords,
                    **robot_positions
                }

                # Display camera frames
                if frames['frame1'] is not None:
                    # Add status info to frame
                    cv2.putText(frames['frame1'], f"Camera 1 - Snapshot #{snapshot_count + 1}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frames['frame1'], "SPACE: Snapshot | ESC: Quit",
                              (10, frames['frame1'].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow('Camera 1', frames['frame1'])

                if frames['frame2'] is not None:
                    cv2.putText(frames['frame2'], f"Camera 2 - Snapshot #{snapshot_count + 1}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Camera 2', frames['frame2'])

                # Show current status in terminal
                hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"
                sync_status = "🔄" if self.sync_active else "⏸️"

                print(f"\r{hand_status} {robot_status} {sync_status} | "
                      f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                      f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                      f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                      f"{data_point['follower_pos3']},{data_point['follower_pos4']}] "
                      f"| Snapshots: {snapshot_count}", end="", flush=True)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Spacebar for snapshot
                    self.record_data_point(data_point)
                    snapshot_count += 1
                    print(f"\n📸 Snapshot {snapshot_count} recorded!")
                elif key == 27:  # ESC key
                    break

        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()
        print(f"\n📸 Snapshot mode ended. Total snapshots: {snapshot_count}")

    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")

        self.recording = False
        self.snapshot_mode = False
        self.running = False

        # Close cameras
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        cv2.destroyAllWindows()

        # Close robot arm connection
        if self.follower_port_handler:
            self.follower_port_handler.closePort()

        # Close output file
        if self.output_file:
            self.output_file.close()

        print("✅ Cleanup complete")

    def start_synchronization(self):
        """Start leader-follower synchronization"""
        if self.sync_active:
            print("⚠️  Synchronization is already running")
            return False

        # Enable follower torques
        if not self.enable_follower_torques():
            print("❌ Failed to enable follower torques")
            return False

        # Start sync thread
        self.sync_active = True
        self.sync_thread = threading.Thread(target=self.sync_leader_to_follower, daemon=True)
        self.sync_thread.start()

        return True

    def stop_synchronization(self):
        """Stop leader-follower synchronization"""
        if not self.sync_active:
            print("⚠️  Synchronization is not running")
            return

        self.sync_active = False
        print("⏳ Stopping synchronization...")
        time.sleep(0.2)

        # Disable follower torques
        self.disable_follower_torques()

def main():
    parser = argparse.ArgumentParser(description="Unified Hand Tracking and Robot Arm Logger")
    parser.add_argument("--mode", choices=['continuous', 'snapshot'], default='continuous',
                       help="Recording mode: continuous or snapshot")
    parser.add_argument("--output", help="Output CSV file name")
    parser.add_argument("--test", action='store_true', help="Test mode - show data without recording")
    parser.add_argument("--sync", action='store_true', help="Enable leader-follower synchronization")
    parser.add_argument("--cal-load", help="Load calibration file for sync")
    args = parser.parse_args()

    logger = UnifiedLogger()

    # Setup signal handler
    def signal_handler(sig, frame):
        logger.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize all systems
        print("🚀 Initializing Unified Logger...")
        print("=" * 60)

        if not logger.init_cameras():
            print("⚠️ Camera initialization failed, continuing with robot only")

        if not logger.init_leader_arm():
            print("⚠️ Leader arm initialization failed, continuing without sync")

        if not logger.init_robot_arm():
            print("⚠️ Follower arm initialization failed, continuing with cameras only")

        # Load calibration if specified
        if args.cal_load:
            logger.load_calibration(args.cal_load)
        elif args.sync:
            logger.load_calibration()  # Try default calibration.json

        if args.test:
            print("🧪 Test mode - showing live data with camera display (no recording)")
            if args.sync:
                print("🔄 Sync mode enabled - leader-follower synchronization active")
                logger.start_synchronization()
            print("🎥 Live camera display with hand tracking enabled")
            print("Press Ctrl+C or ESC to stop")

            try:
                while True:
                    # Get data with frames for display
                    hand_coords, frames = logger.get_hand_coordinates_with_frames(draw_landmarks=True)
                    robot_positions = logger.get_robot_positions()

                    # Combine data
                    data_point = {
                        'timestamp': time.time(),
                        **hand_coords,
                        **robot_positions
                    }

                    # Display camera frames
                    if frames['frame1'] is not None:
                        cv2.putText(frames['frame1'], "Test Mode - No Recording",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frames['frame1'], "Ctrl+C or ESC: Stop",
                                  (10, frames['frame1'].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.imshow('Camera 1 - Test Mode', frames['frame1'])

                    if frames['frame2'] is not None:
                        cv2.putText(frames['frame2'], "Test Mode - No Recording",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow('Camera 2 - Test Mode', frames['frame2'])

                    # Terminal display
                    hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                    robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"
                    sync_status = "🔄" if logger.sync_active else "⏸️"

                    print(f"\r{hand_status} {robot_status} {sync_status} | "
                          f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                          f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                          f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                          f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="", flush=True)

                    # Check for ESC key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
            finally:
                cv2.destroyAllWindows()
                if logger.sync_active:
                    logger.stop_synchronization()
        else:
            # Initialize output file
            output_filename = logger.init_output_file(args.output)

            print("✅ All systems initialized successfully")
            print(f"📊 Data format: timestamp, cam1_x, cam1_y, cam2_x, cam2_y, follower_pos1-4")
            if args.sync:
                print("🔄 Leader-follower synchronization will be active during recording")
            print("=" * 60)

            # Start sync if requested
            if args.sync:
                logger.start_synchronization()

            try:
                # Start recording based on mode
                if args.mode == 'continuous':
                    logger.start_continuous_recording()
                else:
                    logger.start_snapshot_mode()
            finally:
                if logger.sync_active:
                    logger.stop_synchronization()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.cleanup()

if __name__ == "__main__":
    main()