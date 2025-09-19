#!/usr/bin/env python3
"""
Complete logging system startup script
Handles the correct startup sequence to avoid port conflicts
"""

import subprocess
import time
import os
import signal
import sys

def check_process_running(process_name):
    """Check if a process is already running"""
    try:
        result = subprocess.run(['pgrep', '-f', process_name], capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def start_robot_data_server():
    """Start the robot data server"""
    if check_process_running('shared_robot_interface.py'):
        print("📡 Robot data server already running")
        return None

    print("🚀 Starting robot data server...")
    process = subprocess.Popen([
        'python3', 'shared_robot_interface.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait a bit for server to start
    time.sleep(2)

    if process.poll() is None:
        print("✅ Robot data server started")
        return process
    else:
        print("❌ Failed to start robot data server")
        return None

def show_menu():
    """Show startup menu"""
    print("\n" + "="*60)
    print("🎯 Integrated Logging System")
    print("="*60)
    print("1. Start complete system (recommended)")
    print("2. Start robot data server only")
    print("3. Start logger only (server must be running)")
    print("4. Check system status")
    print("5. Stop all processes")
    print("6. Exit")
    print("="*60)

def start_complete_system():
    """Start the complete logging system"""
    print("🚀 Starting complete logging system...")

    # Step 1: Start robot data server
    server_process = start_robot_data_server()
    if not server_process:
        print("❌ Cannot start complete system without robot data server")
        return

    # Step 2: Give instructions for manual steps
    print("\n" + "="*60)
    print("📋 MANUAL STEPS REQUIRED:")
    print("="*60)
    print("1. In ANOTHER TERMINAL, run:")
    print("   python3 leader_follower_sync.py")
    print("   Then type 'start' to begin synchronization")
    print("")
    print("2. Press ENTER here when step 1 is complete...")
    input()

    # Step 3: Start logger
    print("🎥 Starting integrated logger...")
    try:
        subprocess.run(['python3', 'run_logger.py'])
    except KeyboardInterrupt:
        print("\n🛑 Logger stopped")

    # Cleanup
    print("🧹 Stopping robot data server...")
    if server_process:
        server_process.terminate()
        server_process.wait()

def check_system_status():
    """Check status of all system components"""
    print("\n📊 System Status:")
    print("-" * 40)

    # Check robot data server
    if check_process_running('shared_robot_interface.py'):
        print("✅ Robot data server: Running")
    else:
        print("❌ Robot data server: Not running")

    # Check leader_follower_sync
    if check_process_running('leader_follower_sync.py'):
        print("✅ Leader-follower sync: Running")
    else:
        print("❌ Leader-follower sync: Not running")

    # Check integrated logger
    if check_process_running('integrated_logger.py'):
        print("✅ Integrated logger: Running")
    else:
        print("❌ Integrated logger: Not running")

    # Check camera devices
    cameras = []
    for i in range(4):
        if os.path.exists(f"/dev/video{i}"):
            cameras.append(i)

    if cameras:
        print(f"✅ Cameras available: {cameras}")
    else:
        print("❌ No cameras detected")

    # Check robot ports
    robot_ports = []
    for port in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/leader_arm", "/dev/follower_arm"]:
        if os.path.exists(port):
            robot_ports.append(port)

    if robot_ports:
        print(f"✅ Robot ports available: {robot_ports}")
    else:
        print("❌ No robot ports detected")

def stop_all_processes():
    """Stop all related processes"""
    print("🛑 Stopping all processes...")

    processes = [
        'shared_robot_interface.py',
        'leader_follower_sync.py',
        'integrated_logger.py',
        'run_logger.py'
    ]

    for process_name in processes:
        try:
            result = subprocess.run(['pkill', '-f', process_name], capture_output=True)
            if result.returncode == 0:
                print(f"✅ Stopped: {process_name}")
            else:
                print(f"ℹ️  Not running: {process_name}")
        except:
            print(f"❌ Error stopping: {process_name}")

    print("🧹 Cleanup complete")

def main():
    """Main menu loop"""
    print("🎯 Integrated Logging System Launcher")
    print("This script helps avoid port conflicts and manages startup sequence")

    while True:
        show_menu()
        choice = input("\n🎯 Select option (1-6): ").strip()

        if choice == '1':
            start_complete_system()
        elif choice == '2':
            start_robot_data_server()
            print("Robot data server started. Use option 3 to start logger.")
        elif choice == '3':
            print("🎥 Starting logger only...")
            try:
                subprocess.run(['python3', 'run_logger.py'])
            except KeyboardInterrupt:
                print("\n🛑 Logger stopped")
        elif choice == '4':
            check_system_status()
        elif choice == '5':
            stop_all_processes()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

        if choice != '6':
            print("\nPress ENTER to continue...")
            input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)