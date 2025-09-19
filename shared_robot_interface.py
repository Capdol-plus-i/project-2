#!/usr/bin/env python3
"""
Shared Robot Interface for Multiple Clients
Provides robot arm data through socket communication to avoid port conflicts
"""

import socket
import json
import time
import threading
import os
import sys
from dynamixel_sdk import *

class RobotDataServer:
    def __init__(self, port=12345):
        self.port = port
        self.running = False
        self.clients = []
        self.lock = threading.Lock()

        # Robot configuration
        self.load_hardware_config()
        self.init_robot_config()

        # Robot connections
        self.leader_port_handler = None
        self.leader_packet_handler = None
        self.follower_port_handler = None
        self.follower_packet_handler = None
        self.connected_leader_motors = []
        self.connected_follower_motors = []

    def load_hardware_config(self):
        """Load hardware configuration"""
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
        """Initialize robot configuration"""
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

    def init_robot_connection(self):
        """Initialize robot arm connection"""
        print("🤖 Connecting to follower robot arm...")

        # Only connect to follower arm (leader_follower_sync handles leader)
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
            model_number, comm_result, error = self.follower_packet_handler.ping(
                self.follower_port_handler, motor_id)
            if comm_result == COMM_SUCCESS:
                self.connected_follower_motors.append(motor_id)

        print(f"✓ Follower motors: {self.connected_follower_motors}")
        return len(self.connected_follower_motors) > 0

    def get_follower_positions(self):
        """Get current follower positions"""
        positions = {}
        for motor_id in [1, 2, 3, 4]:
            if motor_id in self.connected_follower_motors:
                try:
                    position, comm_result, error = self.follower_packet_handler.read4ByteTxRx(
                        self.follower_port_handler, motor_id, self.ADDR_PRESENT_POSITION)

                    if comm_result == COMM_SUCCESS and error == 0:
                        positions[f'follower_pos{motor_id}'] = position
                    else:
                        positions[f'follower_pos{motor_id}'] = None
                except:
                    positions[f'follower_pos{motor_id}'] = None
            else:
                positions[f'follower_pos{motor_id}'] = None

        return positions

    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        print(f"📡 Client connected: {address}")

        try:
            while self.running:
                # Get robot data
                robot_data = self.get_follower_positions()
                robot_data['timestamp'] = time.time()

                # Send data to client
                data_json = json.dumps(robot_data) + '\n'
                client_socket.send(data_json.encode())

                time.sleep(0.05)  # 20Hz update rate

        except (ConnectionResetError, BrokenPipeError):
            print(f"📡 Client disconnected: {address}")
        except Exception as e:
            print(f"❌ Client error {address}: {e}")
        finally:
            client_socket.close()
            with self.lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)

    def start_server(self):
        """Start the robot data server"""
        if not self.init_robot_connection():
            print("❌ Failed to initialize robot connection")
            return False

        self.running = True

        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server_socket.bind(('localhost', self.port))
            server_socket.listen(5)
            print(f"🌐 Robot data server started on port {self.port}")
            print("📡 Waiting for logger connections...")

            while self.running:
                try:
                    client_socket, address = server_socket.accept()

                    with self.lock:
                        self.clients.append(client_socket)

                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()

                except OSError:
                    break  # Server socket closed

        except Exception as e:
            print(f"❌ Server error: {e}")
        finally:
            server_socket.close()
            self.cleanup()

    def stop_server(self):
        """Stop the server"""
        print("🛑 Stopping robot data server...")
        self.running = False

        # Close all client connections
        with self.lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()

    def cleanup(self):
        """Cleanup resources"""
        if self.follower_port_handler:
            self.follower_port_handler.closePort()
        print("✅ Robot data server stopped")

class RobotDataClient:
    """Client to receive robot data from server"""
    def __init__(self, port=12345):
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """Connect to robot data server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', self.port))
            self.connected = True
            print("📡 Connected to robot data server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to robot data server: {e}")
            return False

    def get_robot_positions(self):
        """Get robot positions from server"""
        if not self.connected:
            return {
                'follower_pos1': None, 'follower_pos2': None,
                'follower_pos3': None, 'follower_pos4': None
            }

        try:
            # Receive data from server - handle line-based protocol
            if not hasattr(self, '_buffer'):
                self._buffer = ""

            # Receive new data
            data = self.socket.recv(1024).decode()
            if not data:
                self.connected = False
                return {
                    'follower_pos1': None, 'follower_pos2': None,
                    'follower_pos3': None, 'follower_pos4': None
                }

            self._buffer += data

            # Process complete lines
            while '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                if line.strip():
                    try:
                        robot_data = json.loads(line.strip())
                        return {
                            'follower_pos1': robot_data.get('follower_pos1'),
                            'follower_pos2': robot_data.get('follower_pos2'),
                            'follower_pos3': robot_data.get('follower_pos3'),
                            'follower_pos4': robot_data.get('follower_pos4')
                        }
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        except Exception as e:
            print(f"⚠️ Error receiving robot data: {e}")
            self.connected = False

        return {
            'follower_pos1': None, 'follower_pos2': None,
            'follower_pos3': None, 'follower_pos4': None
        }

    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
        self.connected = False

def main():
    """Run robot data server"""
    import signal

    server = RobotDataServer()

    def signal_handler(sig, frame):
        server.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("🤖 Starting Robot Data Server...")
    print("This provides robot arm data to multiple clients without port conflicts")
    print("Press Ctrl+C to stop")

    server.start_server()

if __name__ == "__main__":
    main()