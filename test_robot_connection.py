#!/usr/bin/env python3
"""
Test robot arm connection and data retrieval
"""

import json
import os
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

def load_hardware_config():
    """Load hardware configuration"""
    config_file = 'hardware_config.json'
    default_config = {
        'leader': {'port': '/dev/ttyACM0', 'baudrate': 1000000},
        'follower': {'port': '/dev/ttyACM1', 'baudrate': 1000000}
    }

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                robot_config = config.get('robot_arms', {})
                return {
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
            return default_config

    return default_config

def test_robot_connection(config, name):
    """Test connection to a robot arm"""
    print(f"\n🤖 Testing {name} arm connection...")
    print(f"Port: {config['port']}")
    print(f"Baudrate: {config['baudrate']}")

    # Check if port exists
    if not os.path.exists(config['port']):
        print(f"❌ Port {config['port']} does not exist")
        return False

    # Initialize connection
    port_handler = PortHandler(config['port'])
    packet_handler = PacketHandler(2.0)

    if not port_handler.openPort():
        print(f"❌ Failed to open port {config['port']}")
        return False

    if not port_handler.setBaudRate(config['baudrate']):
        print(f"❌ Failed to set baudrate {config['baudrate']}")
        port_handler.closePort()
        return False

    print(f"✅ Port connection successful")

    # Test motor communication
    MOTOR_IDS = [1, 2, 3, 4]
    ADDR_PRESENT_POSITION = 132
    connected_motors = []

    print(f"📡 Scanning for motors...")
    for motor_id in MOTOR_IDS:
        model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
        if comm_result == COMM_SUCCESS:
            print(f"  ✅ Motor {motor_id}: Found (Model: {model_number})")
            connected_motors.append(motor_id)

            # Test reading position
            position, comm_result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION)

            if comm_result == COMM_SUCCESS and error == 0:
                print(f"     Position: {position}")
            else:
                print(f"     ⚠️ Position read failed (comm: {comm_result}, error: {error})")
        else:
            print(f"  ❌ Motor {motor_id}: Not found (comm: {comm_result})")

    port_handler.closePort()

    print(f"📊 Summary: {len(connected_motors)}/{len(MOTOR_IDS)} motors connected")
    return len(connected_motors) > 0

def main():
    print("🔧 Robot Arm Connection Test")
    print("=" * 50)

    # Load configuration
    hw_config = load_hardware_config()
    print(f"📋 Configuration loaded:")
    print(f"  Leader: {hw_config['leader']}")
    print(f"  Follower: {hw_config['follower']}")

    # Test connections
    leader_ok = test_robot_connection(hw_config['leader'], "Leader")
    follower_ok = test_robot_connection(hw_config['follower'], "Follower")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Leader arm: {'✅ OK' if leader_ok else '❌ Failed'}")
    print(f"  Follower arm: {'✅ OK' if follower_ok else '❌ Failed'}")

    if not follower_ok:
        print("\n🔧 Troubleshooting tips for follower arm:")
        print("1. Check power connections")
        print("2. Verify USB cable connections")
        print("3. Check if another program is using the port")
        print("4. Try different port in hardware_config.json")
        print("5. Check motor IDs (should be 1, 2, 3, 4)")

if __name__ == "__main__":
    main()