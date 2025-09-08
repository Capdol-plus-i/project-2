#!/usr/bin/env python3
"""
Configuration module for the robot control system.
Contains all constants, settings, and configuration parameters.
"""

import platform

# Model and file paths
MODEL_PATH = "models/model_parameters_resnet.npz"
CSV_FILENAME = "robot_snapshots.csv"

# Robot hardware configuration
BUTTON_POSITIONS = {0: 2124, 1: 3148}

# ========== HARDWARE CONFIGURATION SECTION ==========
# You can modify these settings for your specific hardware setup

# Arduino Controller Configuration
ARDUINO_CONFIG = {
    'enabled': True,
    'port': 'COM4',
    'baudrate': 9600,
    'auto_detect': True,                # Try to auto-detect Arduino port
    'timeout': 1.0
}

# Robot Arms Configuration
ROBOT_ARMS_CONFIG = {
    'enabled': True,
    'follower': {
        'port': 'COM5',
        'baudrate': 1000000,
        'motor_ids': [1, 2, 3, 4],
        'enabled': True
    },
    'leader': {
        'port': 'COM3',
        'baudrate': 1000000,
        'motor_ids': [1],
        'enabled': True
    },
    'auto_detect': True,                # Try to auto-detect robot arm ports
    'position_p_gain': [200, 200, 400, 200]
}

# Camera Configuration
CAMERA_CONFIG = {
    'enabled': True,                    # Enable/disable cameras
    'camera1': {
        'id': 2,
        'enabled': True,
        'width': 640,
        'height': 480
    },
    'camera2': {
        'id': 3,
        'enabled': True,
        'width': 640,
        'height': 480
    },
    'auto_detect': True,                # Try to auto-detect cameras
    'attempts_read_frame': 1
}

# ========== LEGACY COMPATIBILITY SECTION ==========
# These are maintained for backward compatibility with existing code

# Extract values for legacy variable names
PORT_ACM0 = ROBOT_ARMS_CONFIG['follower']['port']
PORT_ACM1 = ROBOT_ARMS_CONFIG['leader']['port'] 
ARDUINO_PORT = ARDUINO_CONFIG['port']
BAUDRATE = ROBOT_ARMS_CONFIG['follower']['baudrate']
ARDUINO_BAUDRATE = ARDUINO_CONFIG['baudrate']
DYNAMIXEL_IDS_ACM0 = ROBOT_ARMS_CONFIG['follower']['motor_ids']
DYNAMIXEL_IDS_ACM1 = ROBOT_ARMS_CONFIG['leader']['motor_ids']
POSITION_P_GAIN = ROBOT_ARMS_CONFIG['position_p_gain']
CAM_IDS = [CAMERA_CONFIG['camera1']['id'], CAMERA_CONFIG['camera2']['id']]
ATTEMPTS_READ_FRAME_COUNT = CAMERA_CONFIG['attempts_read_frame']

# Default joint positions
DEFAULT_JOINTS = [3000, 2470, 1530, 1530]

# Control timing configuration
HANDS_TIMEOUT = 3.0  # seconds

# Voice recognition configuration
VOICE_RATE = 16000
VOICE_CHUNK = int(VOICE_RATE / 5)  # 100ms chunks
WAKE_WORDS = ["하이봇", "하이못", "아이봇", "AI봇", "로봇아"]

# Voice command mapping
VOICE_CMD_MAP = {
    "시작": ["시작", "시작해", "제어 시작", "컨트롤 시작"],
    "정지": ["정지", "멈춰", "스톱", "중지"],
    "모드0": ["모드 0", "모드0", "모드 영", "모드 제로"],
    "모드1": ["모드 1", "모드1", "모드 일", "모드 원"],
    "밝게": ["밝게", "밝기 올려", "더 밝게", "라이트 업"],
    "어둡게": ["어둡게", "밝기 내려", "어둡게 해", "라이트 다운"],
    "리셋": ["리셋", "초기화", "재설정"],
    "왼쪽": ["왼쪽", "왼 쪽", "왼", "좌측"],
    "오른쪽": ["오른쪽", "오른 쪽", "오른", "우측"],
    "위": ["위", "위로", "위쪽"],
    "아래": ["아래", "아래로", "아레"],
    "종료": ["종료", "끝내", "종료해", "시스템 종료"]
}

# CSV file configuration
CSV_HEADERS = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y", 
               "joint_1", "joint_2", "joint_3", "joint_4"]