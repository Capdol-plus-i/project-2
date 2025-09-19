#!/usr/bin/env python3
"""
Safe robot server starter - handles port conflicts
"""

import subprocess
import time
import socket

def check_port_free(port):
    """Check if port is free"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def kill_existing_servers():
    """Kill existing robot server processes"""
    try:
        result = subprocess.run(['pkill', '-f', 'shared_robot_interface.py'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🧹 Stopped existing robot servers")
            time.sleep(2)  # Wait for cleanup
        return True
    except Exception as e:
        print(f"⚠️ Error killing processes: {e}")
        return False

def main():
    port = 12345

    print("🚀 Starting Robot Data Server Safely...")

    # Check if port is free
    if not check_port_free(port):
        print(f"⚠️ Port {port} is in use")
        print("🧹 Cleaning up existing processes...")
        kill_existing_servers()

        # Check again
        if not check_port_free(port):
            print(f"❌ Port {port} still in use. Please check manually:")
            print(f"   netstat -tulpn | grep :{port}")
            print(f"   kill <PID>")
            return

    print(f"✅ Port {port} is available")
    print("🤖 Starting robot data server...")

    # Start the server
    try:
        subprocess.run(['python3', 'shared_robot_interface.py'])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()