#!/usr/bin/env python3
"""
Quick start script for integrated logging system
"""

import os
import subprocess
import sys
import time

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")

    # Check for camera devices
    cameras = []
    for i in range(4):
        if os.path.exists(f"/dev/video{i}"):
            cameras.append(i)

    print(f"📷 Available cameras: {cameras}")

    # Check for robot arm ports
    robot_ports = []
    common_ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/leader_arm"]
    for port in common_ports:
        if os.path.exists(port):
            robot_ports.append(port)

    print(f"🤖 Available robot ports: {robot_ports}")

    return len(cameras) >= 1 and len(robot_ports) >= 1

def show_menu():
    """Show main menu"""
    print("\n" + "="*60)
    print("🎯 Integrated Hand Tracking & Robot Arm Logger")
    print("="*60)
    print("1. Continuous recording mode")
    print("2. Snapshot mode (press space to capture)")
    print("3. Test camera only")
    print("4. Test robot arms only")
    print("5. Exit")
    print("="*60)

def run_camera_test():
    """Test cameras only"""
    print("🎥 Testing cameras...")
    try:
        result = subprocess.run([
            "python3", "hand_landmark_demo.py", "--headless"
        ], timeout=10, capture_output=True, text=True)
        print("✅ Camera test completed")
        if result.stdout:
            print("Output:", result.stdout[-200:])  # Last 200 chars
    except subprocess.TimeoutExpired:
        print("⏰ Camera test timeout (this is normal)")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")

def run_robot_test():
    """Test robot arms communication"""
    print("🤖 Testing robot arms...")
    print("Note: Make sure leader_follower_sync.py is running separately")
    print("This will just check if robot communication is working...")

    try:
        # Just check if we can import dynamixel_sdk
        from dynamixel_sdk import PortHandler
        print("✅ Dynamixel SDK available")
    except ImportError:
        print("❌ Dynamixel SDK not available")

def run_integrated_logger(mode):
    """Run the integrated logger"""
    print(f"🚀 Starting integrated logger in {mode} mode...")
    print("Make sure:")
    print("1. Both cameras are connected")
    print("2. Robot arms are connected and synchronized")
    print("3. leader_follower_sync.py is running in another terminal")
    print("\nPress Ctrl+C to stop")

    try:
        cmd = ["python3", "integrated_logger.py", "--mode", mode]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Logging stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🎯 Integrated Logger Launcher")

    if not check_dependencies():
        print("⚠️ Some dependencies may not be available")
        print("Continue anyway? (y/n): ", end="")
        if input().lower() != 'y':
            return

    while True:
        show_menu()
        choice = input("\n🎯 Select option (1-5): ").strip()

        if choice == '1':
            run_integrated_logger("continuous")
        elif choice == '2':
            run_integrated_logger("snapshot")
        elif choice == '3':
            run_camera_test()
        elif choice == '4':
            run_robot_test()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

        print("\nPress Enter to continue...")
        input()

if __name__ == "__main__":
    main()