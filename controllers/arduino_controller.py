#!/usr/bin/env python3
"""
Arduino Controller module for managing Arduino communication and LED control.
"""

import os
import time
import threading
import logging
import serial
from config import ARDUINO_BAUDRATE, BUTTON_POSITIONS

logger = logging.getLogger(__name__)

class ArduinoController:
    def __init__(self, port, baudrate=ARDUINO_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.connected = False
        self.running = False
        self.status_lock = threading.Lock()
        
        # Arduino status
        self.arduino_status = {
            'connected': False,
            'brightness_level': 0,
            'robot_mode': 0,
            'last_heartbeat': 0,
            'last_status_update': 0
        }
        
        # Heartbeat settings
        self.heartbeat_interval = 5.0  # seconds
        self.last_heartbeat_sent = 0
        
    def connect(self):
        """Connect to Arduino"""
        if self.connected:
            return True
            
        try:
            # Check port availability (different methods for Windows/Linux)
            import platform
            if platform.system() == "Windows":
                # For Windows COM ports, try to connect directly
                # as os.path.exists() doesn't work for COM ports
                try:
                    test_port = serial.Serial(self.port, self.baudrate, timeout=0.1)
                    test_port.close()
                except serial.SerialException:
                    logger.warning(f"Arduino COM port {self.port} not available")
                    return False
            else:
                # For Linux/macOS, check if device file exists
                if not os.path.exists(self.port):
                    logger.warning(f"Arduino port {self.port} not found")
                    return False
                
            self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino reset delay
            
            logger.info(f"Connected to Arduino on {self.port}")
            self.connected = True
            self.running = True
            
            # Start communication thread
            threading.Thread(target=self._communication_loop, daemon=True).start()
            
            # Send initial heartbeat
            self._send_heartbeat()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def _communication_loop(self):
        """Arduino communication loop"""
        while self.running and self.connected:
            try:
                # Send periodic heartbeat
                current_time = time.time()
                if current_time - self.last_heartbeat_sent >= self.heartbeat_interval:
                    self._send_heartbeat()
                
                # Read incoming data
                if self.serial_port and self.serial_port.in_waiting > 0:
                    try:
                        line = self.serial_port.readline().decode('utf-8').strip()
                        if line:
                            self._process_arduino_message(line)
                    except UnicodeDecodeError:
                        # Clear buffer on decode error
                        self.serial_port.reset_input_buffer()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Arduino communication error: {e}")
                self.connected = False
                break
    
    def _send_heartbeat(self):
        """Send heartbeat to Arduino"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.write(b"HEARTBEAT\n")
                self.last_heartbeat_sent = time.time()
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    def _process_arduino_message(self, message):
        """Process messages from Arduino"""
        try:
            # Handle robot commands (existing functionality)
            if message.startswith("CMD:ROBOT:"):
                mode = int(message.split(":")[-1])
                position = BUTTON_POSITIONS.get(mode)
                if position is not None:
                    # Import socketio here to avoid circular imports
                    from app import robot, socketio
                    if robot and robot.is_arm_connected('leader'):
                        robot.set_joint_position(0, position, 'leader')
                        socketio.emit('robot_mode', {'mode': mode, 'position': position})
                        logger.info(f"Arduino robot command: Mode {mode}, Position {position}")
                        
                        # Update Arduino status
                        with self.status_lock:
                            self.arduino_status['robot_mode'] = mode
                            
            # Handle LED commands
            elif message.startswith("CMD:LED:"):
                level = int(message.split(":")[-1])
                logger.info(f"Arduino LED command: Level {level}")
                from app import socketio
                socketio.emit('arduino_led', {'brightness_level': level})
                
                # Update Arduino status
                with self.status_lock:
                    self.arduino_status['brightness_level'] = level
                    
            # Handle status updates
            elif message.startswith("STATUS:"):
                self._parse_status_message(message)
                
            # Handle heartbeat acknowledgment
            elif message.startswith("ACK:HEARTBEAT"):
                with self.status_lock:
                    self.arduino_status['last_heartbeat'] = time.time()
                    self.arduino_status['connected'] = True
                    
            # Log other messages
            else:
                logger.info(f"Arduino: {message}")
                
        except Exception as e:
            logger.error(f"Error processing Arduino message '{message}': {e}")
    
    def _parse_status_message(self, message):
        """Parse Arduino status message: STATUS:BRIGHTNESS:X:MODE:X:CONNECTED:X"""
        try:
            parts = message.split(':')
            if len(parts) >= 7:
                brightness = int(parts[2])
                mode = int(parts[4])
                connected = int(parts[6])
                
                with self.status_lock:
                    self.arduino_status.update({
                        'brightness_level': brightness,
                        'robot_mode': mode,
                        'connected': bool(connected),
                        'last_status_update': time.time()
                    })
                
                # Emit status to web clients
                from app import socketio
                socketio.emit('arduino_status', self.arduino_status.copy())
                
        except Exception as e:
            logger.error(f"Error parsing Arduino status: {e}")
    
    def send_command(self, command):
        """Send command to Arduino"""
        try:
            if self.serial_port and self.serial_port.is_open:
                cmd = f"{command}\n"
                self.serial_port.write(cmd.encode('utf-8'))
                logger.info(f"Sent to Arduino: {command}")
                return True
        except Exception as e:
            logger.error(f"Failed to send Arduino command '{command}': {e}")
        return False
    
    def set_brightness(self, level):
        """Set Arduino LED brightness level (0-5)"""
        if 0 <= level <= 5:
            return self.send_command(f"SET_BRIGHTNESS:{level}")
        return False
    
    def set_robot_mode(self, mode):
        """Set Arduino robot mode (0 or 1)"""
        if mode in [0, 1]:
            return self.send_command(f"SET_MODE:{mode}")
        return False
    
    def trigger_led_effect(self, effect_type):
        """Trigger LED effect (0: none, 1: flash, 2: blue fade, 3: green fade)"""
        if 0 <= effect_type <= 3:
            return self.send_command(f"LED_EFFECT:{effect_type}")
        return False
    
    def reset_arduino(self):
        """Reset Arduino to default state"""
        return self.send_command("RESET")
    
    def get_status(self):
        """Get current Arduino status"""
        with self.status_lock:
            return self.arduino_status.copy()
    
    def is_connected(self):
        """Check if Arduino is connected"""
        return self.connected and self.arduino_status.get('connected', False)
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.reset_arduino()
        self.running = False
        self.connected = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
        logger.info("Disconnected from Arduino")