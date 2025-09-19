#!/usr/bin/env python3
"""
Integrated Hand Tracking and Robot Arm Logger
Records hand landmarks from dual cameras and robot arm positions simultaneously
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

# Hand tracking imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Robot arm imports
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

class IntegratedLogger:
    def __init__(self):
        self.recording = False
        self.snapshot_mode = False
        self.data_buffer = []
        self.lock = threading.Lock()

        # Camera settings
        self.DEV1 = os.environ.get("DEV", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0")
        self.DEV2 = os.environ.get("DEV2", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0")
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

        # Robot arm client
        self.robot_client = None

        # Output file
        self.output_file = None
        self.csv_writer = None

    def load_hardware_config(self):
        """Load hardware configuration from hardware_config.json"""
        config_file = 'hardware_config.json'
        default_config = {
            'leader': {'port': '/dev/leader_arm', 'baudrate': 1000000},
            'follower': {'port': '/dev/ttyACM2', 'baudrate': 1000000}
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
        """Initialize robot arm communication via shared interface"""
        print("🤖 Connecting to robot data server...")

        # Import and create robot client
        from shared_robot_interface import RobotDataClient
        self.robot_client = RobotDataClient()

        if self.robot_client.connect():
            print("✓ Connected to robot data server")
            return True
        else:
            print("❌ Failed to connect to robot data server")
            print("Make sure shared_robot_interface.py is running first!")
            self.robot_client = None
            return False


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
        """Get current positions from robot data server"""
        if hasattr(self, 'robot_client') and self.robot_client:
            return self.robot_client.get_robot_positions()
        else:
            return {
                'follower_pos1': None, 'follower_pos2': None,
                'follower_pos3': None, 'follower_pos4': None
            }

    def init_output_file(self, filename=None):
        """Initialize CSV output file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_log_{timestamp}.csv"

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
        print("📍 Press 's' to take snapshot, 'q' to stop recording")

        self.recording = True
        count = 0

        try:
            while self.recording:
                data_point = self.collect_data_point()
                self.record_data_point(data_point)
                count += 1

                if count % 10 == 0:  # Print status every 10 samples
                    print(f"\r📊 Recorded {count} data points | "
                          f"Cam1: ({data_point['cam1_x']},{data_point['cam1_y']}) | "
                          f"Cam2: ({data_point['cam2_x']},{data_point['cam2_y']}) | "
                          f"Robot: [{data_point['follower_pos1']},{data_point['follower_pos2']},"
                          f"{data_point['follower_pos3']},{data_point['follower_pos4']}]", end="")

                time.sleep(0.1)  # 10Hz sampling rate

        except KeyboardInterrupt:
            self.recording = False

        print(f"\n🛑 Recording stopped. Total: {count} data points")

    def start_snapshot_mode(self):
        """Start snapshot mode - record on demand"""
        print("📸 Snapshot mode activated")
        print("📍 Press SPACE to take snapshot, 'q' to quit")

        self.snapshot_mode = True
        snapshot_count = 0

        # Create a simple window for key capture
        cv2.namedWindow("Snapshot Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Snapshot Control", 400, 100)

        try:
            while self.snapshot_mode:
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # Spacebar for snapshot
                    data_point = self.collect_data_point()
                    self.record_data_point(data_point)
                    snapshot_count += 1

                    print(f"📸 Snapshot {snapshot_count}: "
                          f"Cam1: ({data_point['cam1_x']},{data_point['cam1_y']}) | "
                          f"Cam2: ({data_point['cam2_x']},{data_point['cam2_y']}) | "
                          f"Robot: [{data_point['follower_pos1']},{data_point['follower_pos2']},"
                          f"{data_point['follower_pos3']},{data_point['follower_pos4']}]")

                elif key == 27:  # ESC key
                    break

                time.sleep(0.01)

        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()
        print(f"\n📸 Snapshot mode ended. Total snapshots: {snapshot_count}")

    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")

        self.recording = False
        self.snapshot_mode = False

        # Close cameras
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        cv2.destroyAllWindows()

        # Close robot arm connections
        if hasattr(self, 'robot_client') and self.robot_client:
            self.robot_client.disconnect()

        # Close output file
        if self.output_file:
            self.output_file.close()

        print("✅ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="Integrated Hand Tracking and Robot Arm Logger")
    parser.add_argument("--mode", choices=['continuous', 'snapshot'], default='continuous',
                       help="Recording mode: continuous or snapshot")
    parser.add_argument("--output", help="Output CSV file name")
    args = parser.parse_args()

    logger = IntegratedLogger()

    # Setup signal handler
    def signal_handler(sig, frame):
        logger.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize all systems
        print("🚀 Initializing Integrated Logger...")

        if not logger.init_cameras():
            print("❌ Camera initialization failed")
            return

        if not logger.init_robot_arms():
            print("❌ Robot arm initialization failed")
            return

        # Initialize output file
        output_filename = logger.init_output_file(args.output)

        print("✅ All systems initialized successfully")
        print(f"📊 Data format: timestamp, cam1_x, cam1_y, cam2_x, cam2_y, follower_pos1-4")
        print("-" * 80)

        # Start recording based on mode
        if args.mode == 'continuous':
            logger.start_continuous_recording()
        else:
            # Import numpy for snapshot mode
            try:
                import numpy as np
                logger.start_snapshot_mode()
            except ImportError:
                print("❌ NumPy required for snapshot mode")
                return

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        logger.cleanup()

if __name__ == "__main__":
    main()