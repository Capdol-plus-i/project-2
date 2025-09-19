#!/usr/bin/env python3
"""
Debug robot arm access issues
"""

import time
import json
import os
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

def load_hardware_config():
    """Load hardware configuration"""
    config_file = 'hardware_config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
        robot_config = config.get('robot_arms', {})
        return {
            'leader': robot_config.get('leader', {}),
            'follower': robot_config.get('follower', {})
        }

def test_single_connection():
    """Test connecting to one arm at a time"""
    hw_config = load_hardware_config()

    # Test follower only first
    print("🧪 Testing follower arm only...")

    follower_port = hw_config['follower']['port']
    follower_baudrate = hw_config['follower']['baudrate']

    port_handler = PortHandler(follower_port)
    packet_handler = PacketHandler(2.0)

    if not port_handler.openPort():
        print(f"❌ Failed to open {follower_port}")
        return False

    if not port_handler.setBaudRate(follower_baudrate):
        print(f"❌ Failed to set baudrate {follower_baudrate}")
        return False

    print(f"✅ Connected to {follower_port}")

    # Test reading positions multiple times
    ADDR_PRESENT_POSITION = 132
    MOTOR_IDS = [1, 2, 3, 4]

    for i in range(10):
        print(f"\nTest {i+1}:")
        for motor_id in MOTOR_IDS:
            try:
                position, comm_result, error = packet_handler.read4ByteTxRx(
                    port_handler, motor_id, ADDR_PRESENT_POSITION)

                if comm_result == COMM_SUCCESS and error == 0:
                    print(f"  Motor {motor_id}: {position}")
                else:
                    print(f"  Motor {motor_id}: Failed (comm: {comm_result}, error: {error})")
            except Exception as e:
                print(f"  Motor {motor_id}: Exception - {e}")

        time.sleep(0.5)

    port_handler.closePort()
    print("\n✅ Test completed")

def test_enable_torque():
    """Test enabling torque"""
    hw_config = load_hardware_config()

    print("🧪 Testing torque enable/disable...")

    follower_port = hw_config['follower']['port']
    follower_baudrate = hw_config['follower']['baudrate']

    port_handler = PortHandler(follower_port)
    packet_handler = PacketHandler(2.0)

    if not port_handler.openPort():
        print(f"❌ Failed to open {follower_port}")
        return False

    if not port_handler.setBaudRate(follower_baudrate):
        print(f"❌ Failed to set baudrate {follower_baudrate}")
        return False

    ADDR_TORQUE_ENABLE = 64
    ADDR_PRESENT_POSITION = 132
    MOTOR_IDS = [1, 2, 3, 4]

    # Read positions before torque enable
    print("📊 Positions before torque enable:")
    for motor_id in MOTOR_IDS:
        position, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_POSITION)
        if comm_result == COMM_SUCCESS and error == 0:
            print(f"  Motor {motor_id}: {position}")

    # Enable torques
    print("\n⚡ Enabling torques...")
    for motor_id in MOTOR_IDS:
        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        if comm_result == COMM_SUCCESS and error == 0:
            print(f"  ✅ Motor {motor_id}: Torque enabled")
        else:
            print(f"  ❌ Motor {motor_id}: Torque enable failed (comm: {comm_result}, error: {error})")

    time.sleep(1)

    # Read positions after torque enable
    print("\n📊 Positions after torque enable:")
    for motor_id in MOTOR_IDS:
        position, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_POSITION)
        if comm_result == COMM_SUCCESS and error == 0:
            print(f"  Motor {motor_id}: {position}")
        else:
            print(f"  Motor {motor_id}: Read failed (comm: {comm_result}, error: {error})")

    # Test continuous reading for 5 seconds
    print("\n📊 Continuous reading test (5 seconds)...")
    start_time = time.time()
    read_count = 0
    error_count = 0

    while time.time() - start_time < 5:
        for motor_id in MOTOR_IDS:
            position, comm_result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION)
            if comm_result == COMM_SUCCESS and error == 0:
                read_count += 1
            else:
                error_count += 1
        time.sleep(0.05)

    print(f"  Successful reads: {read_count}")
    print(f"  Failed reads: {error_count}")
    print(f"  Success rate: {read_count/(read_count+error_count)*100:.1f}%")

    # Disable torques
    print("\n⚡ Disabling torques...")
    for motor_id in MOTOR_IDS:
        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        if comm_result == COMM_SUCCESS and error == 0:
            print(f"  ✅ Motor {motor_id}: Torque disabled")
        else:
            print(f"  ❌ Motor {motor_id}: Torque disable failed")

    port_handler.closePort()
    print("\n✅ Torque test completed")

def main():
    print("🔧 Robot Access Debug Tool")
    print("=" * 50)

    print("\n1. Testing single connection...")
    test_single_connection()

    print("\n2. Testing torque enable/disable...")
    test_enable_torque()

if __name__ == "__main__":
    main()