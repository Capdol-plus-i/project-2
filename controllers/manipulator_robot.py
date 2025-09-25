#!/usr/bin/env python3
"""
Manipulator Robot module for controlling Dynamixel servos and robot arm movements.
"""

import os
import logging
from config import PORT_ACM0, PORT_ACM1, BAUDRATE, DYNAMIXEL_IDS_ACM0, DYNAMIXEL_IDS_ACM1, POSITION_P_GAIN

logger = logging.getLogger(__name__)

# Helper functions for Dynamixel SDK
def DXL_LOBYTE(value): return value & 0xFF
def DXL_HIBYTE(value): return (value >> 8) & 0xFF
def DXL_LOWORD(value): return value & 0xFFFF
def DXL_HIWORD(value): return (value >> 16) & 0xFFFF

class ManipulatorRobot:
    def __init__(self):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
            
            self.PortHandler, self.PacketHandler = PortHandler, PacketHandler
            self.GroupSyncWrite, self.GroupSyncRead = GroupSyncWrite, GroupSyncRead
            
            # Protocol and address settings
            self.PROTOCOL_VERSION = 2.0
            self.ADDR_TORQUE_ENABLE = 64
            self.ADDR_OPERATING_MODE = 11
            self.ADDR_GOAL_POSITION = 116
            self.ADDR_PRESENT_POSITION = 132
            self.ADDR_POSITION_P_GAIN = 84
            self.TORQUE_ENABLE, self.TORQUE_DISABLE = 1, 0
            
            self.port_handlers = {
                'follower': self.PortHandler(PORT_ACM0),
                'leader': self.PortHandler(PORT_ACM1)
            }
            self.packet_handlers = {
                'follower': self.PacketHandler(self.PROTOCOL_VERSION),
                'leader': self.PacketHandler(self.PROTOCOL_VERSION)
            }
            self.motor_ids = {
                'follower': DYNAMIXEL_IDS_ACM0,
                'leader': DYNAMIXEL_IDS_ACM1
            }
            self.sync_writers, self.sync_readers = {}, {}
            # Track connection status for each arm
            self.connected_arms = set()
            self.is_connected = False
        except ImportError as e:
            logger.error(f"Dynamixel SDK import error: {e}")
            self.connected_arms = set()
            self.is_connected = False

    def connect(self):
        """Modified to allow partial connections - at least one port must connect"""
        if self.is_connected: return True
        
        connected_count = 0
        self.connected_arms.clear()
        
        try:
            for arm_type, port_handler in self.port_handlers.items():
                try:
                    port_name = port_handler.getPortName()
                    
                    # Check if port exists before trying to connect (platform-specific)
                    import platform
                    if platform.system() == "Windows":
                        # For Windows COM ports, skip existence check and try direct connection
                        # as os.path.exists() doesn't work for COM ports
                        pass
                    else:
                        # For Linux/macOS, check if device file exists
                        if not os.path.exists(port_name):
                            logger.warning(f"Port {port_name} for {arm_type} does not exist, skipping...")
                            continue
                    
                    if not port_handler.openPort():
                        logger.warning(f"Failed to open port {port_name} for {arm_type}")
                        continue
                        
                    if not port_handler.setBaudRate(BAUDRATE):
                        logger.warning(f"Failed to set baudrate for {arm_type}")
                        port_handler.closePort()
                        continue
                    
                    logger.info(f"Connected '{arm_type}' on {port_name}")
                    self.connected_arms.add(arm_type)
                    connected_count += 1
                    
                    # Initialize sync instances only for connected arms
                    self.sync_writers[arm_type] = self.GroupSyncWrite(
                        port_handler, self.packet_handlers[arm_type], 
                        self.ADDR_GOAL_POSITION, 4)
                    self.sync_readers[arm_type] = self.GroupSyncRead(
                        port_handler, self.packet_handlers[arm_type],
                        self.ADDR_PRESENT_POSITION, 4)
                    
                    # Add parameters for sync read
                    for dxl_id in self.motor_ids[arm_type]:
                        self.sync_readers[arm_type].addParam(dxl_id)
                        
                except Exception as e:
                    logger.warning(f"Error connecting {arm_type}: {e}")
                    continue
            
            # Consider connection successful if at least one arm connects
            if connected_count > 0:
                self.is_connected = True
                logger.info(f"Robot partially connected: {connected_count} arm(s) connected - {list(self.connected_arms)}")
                if 'follower' not in self.connected_arms:
                    logger.warning("Follower arm not connected - gesture control will be limited")
                if 'leader' not in self.connected_arms:
                    logger.warning("Leader arm not connected - some manual controls may not work")
                return True
            else:
                logger.error("No robot arms could be connected")
                self._cleanup_connections()
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._cleanup_connections()
            return False

    def _cleanup_connections(self):
        for arm_type, port_handler in self.port_handlers.items():
            try: 
                port_handler.closePort()
            except: 
                pass
        self.connected_arms.clear()
        self.is_connected = False

    def is_arm_connected(self, arm_type):
        """Check if specific arm is connected"""
        return arm_type in self.connected_arms

    def disable_torque(self, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            logger.warning(f"Cannot disable torque for {arm_type} - not connected")
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            for dxl_id in self.motor_ids[arm_type]:
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            logger.info(f"Torque disabled for {arm_type}")
            return True
        except Exception as e:
            logger.error(f"Torque disable error: {e}")
            return False

    def setup_control(self, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            logger.warning(f"Cannot setup control for {arm_type} - not connected")
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            
            for i, dxl_id in enumerate(self.motor_ids[arm_type]):
                # Disable torque to change operating mode
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
                # Set operating mode (Position Control Mode)
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_OPERATING_MODE, 3)
                # Enable torque
                packet_handler.write1ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
                # Set Position P Gain
                packet_handler.write2ByteTxRx(port_handler, dxl_id, 
                                             self.ADDR_POSITION_P_GAIN, POSITION_P_GAIN[i])
            logger.info(f"Control setup complete for {arm_type}")
            return True
        except Exception as e:
            logger.error(f"Control setup error: {e}")
            return False

    def move(self, positions, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms:
            return False
        try:
            sync_writer = self.sync_writers[arm_type]
            sync_writer.clearParam()

            # Pre-calculate all positions for faster processing
            param_list = []
            for i, dxl_id in enumerate(self.motor_ids[arm_type]):
                if i < len(positions):
                    position = max(1, min(4094, int(positions[i])))  # Inline clamp
                    param_goal_position = [
                        position & 0xFF,
                        (position >> 8) & 0xFF,
                        (position >> 16) & 0xFF,
                        (position >> 24) & 0xFF
                    ]
                    param_list.append((dxl_id, param_goal_position))

            # Batch add parameters
            for dxl_id, param in param_list:
                sync_writer.addParam(dxl_id, param)

            # Send without waiting for response (faster)
            result = sync_writer.txPacket()
            return result == 0  # COMM_SUCCESS
        except Exception as e:
            logger.error(f"Move error: {e}")
            return False

    def set_joint_position(self, joint_index, position, arm_type='leader'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            logger.warning(f"Cannot set joint position for {arm_type} - not connected")
            return False
        if joint_index >= len(self.motor_ids[arm_type]): 
            return False
        try:
            port_handler = self.port_handlers[arm_type]
            packet_handler = self.packet_handlers[arm_type]
            dxl_id = self.motor_ids[arm_type][joint_index]
            
            # Clamp position to safe range
            position = max(1, min(4094, int(position)))
            
            dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(
                port_handler, dxl_id, self.ADDR_GOAL_POSITION, position)
            
            if dxl_comm_result != 0 or dxl_error != 0:
                logger.error(f"Joint position error: comm={dxl_comm_result}, error={dxl_error}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Joint position error: {e}")
            return False

    def get_positions(self, arm_type='follower'):
        if not self.is_connected or arm_type not in self.connected_arms: 
            return [None] * len(self.motor_ids[arm_type])
        try:
            sync_reader = self.sync_readers[arm_type]
            
            if sync_reader.txRxPacket() != 0:
                return [None] * len(self.motor_ids[arm_type])
            
            positions = []
            for dxl_id in self.motor_ids[arm_type]:
                if sync_reader.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, 4):
                    positions.append(sync_reader.getData(dxl_id, self.ADDR_PRESENT_POSITION, 4))
                else:
                    positions.append(None)
            
            return positions
        except Exception as e:
            logger.error(f"Position read error: {e}")
            return [None] * len(self.motor_ids[arm_type])

    def disconnect(self):
        if not self.is_connected: return
        for arm_type in self.connected_arms.copy():
            try:
                self.disable_torque(arm_type)
                self.port_handlers[arm_type].closePort()
            except: 
                pass
        self.connected_arms.clear()
        self.is_connected = False
        logger.info("Robot disconnected")