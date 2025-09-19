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

    def load_hardware_config(self):
        """Load hardware configuration from hardware_config.json"""
        config_file = 'hardware_config.json'
        default_config = {
            'follower': {'port': '/dev/follower_arm', 'baudrate': 1000000}
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    robot_config = config.get('robot_arms', {})
                    self.hw_config = {
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
        self.MOTOR_IDS = [1, 2, 3, 4]

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

                if count % 10 == 0:  # Print status every 10 samples
                    hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                    robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"

                    print(f"\r📊 [{count:4d}] {hand_status} Hands | {robot_status} Robot | "
                          f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                          f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                          f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                          f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="", flush=True)

                time.sleep(0.1)  # 10Hz sampling rate

        except KeyboardInterrupt:
            self.recording = False

        print(f"\n🛑 Recording stopped. Total: {count} data points")

    def start_snapshot_mode(self):
        """Start snapshot mode - record on demand"""
        print("📸 Snapshot mode activated")
        print("📍 Press SPACE to take snapshot, ESC to quit")

        self.snapshot_mode = True
        snapshot_count = 0

        try:
            while self.snapshot_mode and self.running:
                # Get current data for preview
                data_point = self.collect_data_point()

                # Show current status
                hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"

                print(f"\r{hand_status} {robot_status} | "
                      f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                      f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                      f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                      f"{data_point['follower_pos3']},{data_point['follower_pos4']}] "
                      f"| Press SPACE for snapshot #{snapshot_count + 1}", end="", flush=True)

                # Check for user input (non-blocking)
                import select
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    if key == ' ':  # Spacebar for snapshot
                        self.record_data_point(data_point)
                        snapshot_count += 1
                        print(f"\n📸 Snapshot {snapshot_count} recorded!")
                    elif key == '\x1b':  # ESC key
                        break

                time.sleep(0.1)

        except KeyboardInterrupt:
            pass

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

def main():
    parser = argparse.ArgumentParser(description="Unified Hand Tracking and Robot Arm Logger")
    parser.add_argument("--mode", choices=['continuous', 'snapshot'], default='continuous',
                       help="Recording mode: continuous or snapshot")
    parser.add_argument("--output", help="Output CSV file name")
    parser.add_argument("--test", action='store_true', help="Test mode - show data without recording")
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

        if not logger.init_robot_arm():
            print("⚠️ Robot arm initialization failed, continuing with cameras only")

        if args.test:
            print("🧪 Test mode - showing live data (no recording)")
            print("Press Ctrl+C to stop")

            while True:
                data_point = logger.collect_data_point()
                hand_status = "👋" if (data_point['cam1_x'] or data_point['cam2_x']) else "🚫"
                robot_status = "🤖" if any(data_point[f'follower_pos{i}'] for i in range(1,5)) else "❌"

                print(f"\r{hand_status} {robot_status} | "
                      f"C1:({data_point['cam1_x']},{data_point['cam1_y']}) "
                      f"C2:({data_point['cam2_x']},{data_point['cam2_y']}) "
                      f"R:[{data_point['follower_pos1']},{data_point['follower_pos2']},"
                      f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="", flush=True)
                time.sleep(0.2)
        else:
            # Initialize output file
            output_filename = logger.init_output_file(args.output)

            print("✅ All systems initialized successfully")
            print(f"📊 Data format: timestamp, cam1_x, cam1_y, cam2_x, cam2_y, follower_pos1-4")
            print("=" * 60)

            # Start recording based on mode
            if args.mode == 'continuous':
                logger.start_continuous_recording()
            else:
                logger.start_snapshot_mode()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.cleanup()

if __name__ == "__main__":
    main()