#!/usr/bin/env python3
"""
Test client for robot data server
"""

import time
import sys
from shared_robot_interface import RobotDataClient

def main():
    print("🧪 Testing robot data client...")

    client = RobotDataClient()

    print("📡 Connecting to robot data server...")
    if not client.connect():
        print("❌ Failed to connect. Make sure shared_robot_interface.py is running")
        return

    print("✅ Connected! Getting robot data...")

    try:
        for i in range(10):
            data = client.get_robot_positions()
            print(f"Sample {i+1}: {data}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Test stopped")

    finally:
        client.disconnect()
        print("📡 Disconnected")

if __name__ == "__main__":
    main()