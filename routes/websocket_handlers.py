#!/usr/bin/env python3
"""
WebSocket event handlers module for Socket.IO communication.
"""

import logging
from datetime import datetime
from flask import request
from flask_socketio import emit
from utils.csv_utils import get_csv_stats, save_snapshot_to_csv
from config import BUTTON_POSITIONS

logger = logging.getLogger(__name__)

def register_websocket_handlers(socketio):
    """Register all WebSocket event handlers with the SocketIO instance"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        
        # Import here to avoid circular imports - get fresh references
        import sys
        app_module = sys.modules.get('app')
        if not app_module:
            return
        
        system_status_data = getattr(app_module, 'system_status_data', {'status': 'initializing'})
        robot_status_data = getattr(app_module, 'robot_status_data', {'connected_arms': []})
        system_initialized = getattr(app_module, 'system_initialized', False)
        controller = getattr(app_module, 'controller', None)
        robot = getattr(app_module, 'robot', None)
        arduino_controller = getattr(app_module, 'arduino_controller', None)
        
        # Send current system status
        socketio.emit('system_status', system_status_data, to=request.sid)
        
        # Send robot connection status
        if robot and robot.is_connected:
            socketio.emit('robot_status', robot_status_data, to=request.sid)
        else:
            socketio.emit('robot_status', {'connected_arms': []}, to=request.sid)
        
        # Send control status
        if controller and controller.running:
            socketio.emit('control_status', {'active': controller.control_active}, to=request.sid)
        else:
            socketio.emit('control_status', {'active': False}, to=request.sid)
        
        # Send Arduino status
        if arduino_controller:
            socketio.emit('arduino_status', arduino_controller.get_status(), to=request.sid)

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('start_control')
    def handle_start_control(data=None):
        logger.info("Received start_control command")
        
        # Import here to avoid circular imports - get fresh references
        import sys
        app_module = sys.modules.get('app')
        if not app_module:
            return {'success': False, 'error': 'App module not available'}
        
        system_initialized = getattr(app_module, 'system_initialized', False)
        controller = getattr(app_module, 'controller', None)
        robot = getattr(app_module, 'robot', None)
        arduino_controller = getattr(app_module, 'arduino_controller', None)
        
        # Detailed debug logging
        logger.info(f"Start control debug - system_initialized: {system_initialized}, controller: {controller is not None}, controller.running: {controller.running if controller else False}")
        if robot:
            logger.info(f"Robot arms: {list(robot.connected_arms)}, follower connected: {robot.is_arm_connected('follower')}")
        
        if not system_initialized:
            return {'success': False, 'error': 'System not initialized'}
        if not controller:
            return {'success': False, 'error': 'Controller not available'}
        if not controller.running:
            return {'success': False, 'error': 'Controller not running'}
        if not robot.is_arm_connected('follower'):
            return {'success': False, 'error': 'Follower arm not connected'}
            
        success = controller.start_control()
        logger.info(f"Start control result: {success}")
        
        if success:
            socketio.emit('control_status', {'active': True})
            # Trigger Arduino LED effect
            if arduino_controller and arduino_controller.is_connected():
                arduino_controller.trigger_led_effect(1)  # Flash effect
        return {'success': success}

    @socketio.on('stop_control')
    def handle_stop_control(data=None):
        logger.info("Received stop_control command")
        
        # Import here to avoid circular imports
        from app import system_initialized, controller, arduino_controller
        
        if not system_initialized or not controller or not controller.running:
            return {'success': False, 'error': 'System not ready'}
        success = controller.pause_control()
        if success:
            socketio.emit('control_status', {'active': False})
            # Trigger Arduino LED effect
            if arduino_controller and arduino_controller.is_connected():
                arduino_controller.trigger_led_effect(0)  # No effect (normal)
        return {'success': success}

    @socketio.on('set_robot_mode')
    def handle_set_robot_mode(data):
        logger.info(f"Received set_robot_mode command: {data}")
        
        # Import here to avoid circular imports
        from app import system_initialized, controller, robot, arduino_controller
        
        if not system_initialized or not controller or not controller.running:
            return {'success': False, 'error': 'System not ready'}
        
        mode = data.get('mode', 0)
        position = BUTTON_POSITIONS.get(mode)
        
        if position is None:
            return {'success': False, 'error': f'Invalid mode: {mode}'}
        
        # Try Arduino first, then robot
        arduino_success = False
        robot_success = False
        
        if arduino_controller and arduino_controller.is_connected():
            arduino_success = arduino_controller.set_robot_mode(mode)
        
        if robot.is_arm_connected('leader'):
            robot_success = robot.set_joint_position(0, position, 'leader')
        
        if arduino_success or robot_success:
            socketio.emit('robot_mode', {'mode': mode, 'position': position})
            return {'success': True, 'mode': mode, 'position': position}
        else:
            return {'success': False, 'error': 'No control interfaces available'}

    # New Arduino WebSocket handlers
    @socketio.on('arduino_set_brightness')
    def handle_arduino_set_brightness(data):
        """Set Arduino LED brightness"""
        # Import here to avoid circular imports
        from app import arduino_controller
        
        if not arduino_controller or not arduino_controller.is_connected():
            return {'success': False, 'error': 'Arduino not connected'}
        
        level = data.get('level', 0)
        if not isinstance(level, int) or not (0 <= level <= 5):
            return {'success': False, 'error': 'Invalid brightness level (0-5)'}
        
        success = arduino_controller.set_brightness(level)
        return {'success': success}

    @socketio.on('arduino_led_effect')
    def handle_arduino_led_effect(data):
        """Trigger Arduino LED effect"""
        # Import here to avoid circular imports
        from app import arduino_controller
        
        if not arduino_controller or not arduino_controller.is_connected():
            return {'success': False, 'error': 'Arduino not connected'}
        
        effect = data.get('effect', 0)
        if not isinstance(effect, int) or not (0 <= effect <= 3):
            return {'success': False, 'error': 'Invalid effect type (0-3)'}
        
        success = arduino_controller.trigger_led_effect(effect)
        return {'success': success}

    @socketio.on('arduino_reset')
    def handle_arduino_reset(data=None):
        """Reset Arduino to default state"""
        # Import here to avoid circular imports
        from app import arduino_controller
        
        if not arduino_controller or not arduino_controller.is_connected():
            return {'success': False, 'error': 'Arduino not connected'}
        
        success = arduino_controller.reset_arduino()
        return {'success': success}

    @socketio.on('get_status')
    def handle_get_status(data=None):
        """Get current system status via WebSocket"""
        # Import here to avoid circular imports
        from app import system_initialized, controller, robot, arduino_controller
        
        arduino_status = arduino_controller.get_status() if arduino_controller else {}
        
        if system_initialized and controller and controller.running:
            return {
                'system_ready': True,
                'control_active': controller.control_active,
                'robot_connected': robot.is_connected,
                'connected_arms': list(robot.connected_arms),
                'hand_detected': controller.hand_detected,
                'last_data': controller.get_last_data(),
                'arduino': arduino_status
            }
        else:
            return {
                'system_ready': False,
                'control_active': False,
                'robot_connected': robot.is_connected if robot else False,
                'connected_arms': list(robot.connected_arms) if robot else [],
                'hand_detected': [False, False],
                'last_data': [None] * 8,
                'arduino': arduino_status
            }

    @socketio.on('take_snapshot')
    def handle_take_snapshot(data=None):
        """Take snapshot of current system state and save to CSV"""
        logger.info("Received take_snapshot command")
        
        # Import here to avoid circular imports
        from app import controller, robot, arduino_controller
        
        if not controller or not controller.running:
            return {'success': False, 'error': 'System not ready'}
        
        try:
            # Get current data
            current_data = controller.get_last_data()
            
            # Validate data (ensure we have all 8 values)
            if len(current_data) != 8:
                return {'success': False, 'error': 'Invalid data length'}
            
            # Save to CSV
            success, total_snapshots = save_snapshot_to_csv(current_data)
            
            if success:
                # Trigger Arduino LED effect for snapshot feedback
                if arduino_controller and arduino_controller.is_connected():
                    arduino_controller.trigger_led_effect(2)  # Blue fade effect
                
                # Emit success event to all clients
                socketio.emit('snapshot_saved', {
                    'success': True,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data': current_data,
                    'filename': 'robot_snapshots.csv',
                    'total_snapshots': total_snapshots,
                    'connected_arms': list(robot.connected_arms) if robot else []
                })
                return {'success': True, 'total_snapshots': total_snapshots}
            else:
                return {'success': False, 'error': 'Failed to save to CSV'}
                
        except Exception as e:
            error_msg = f"Snapshot error: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    @socketio.on('get_csv_stats')
    def handle_get_csv_stats(data=None):
        """Get CSV file statistics"""
        try:
            stats = get_csv_stats()
            return {'success': True, 'stats': stats}
        except Exception as e:
            return {'success': False, 'error': str(e)}