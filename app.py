#!/usr/bin/env python3
import os, time, threading, logging, signal, sys
from flask import Flask
from flask_socketio import SocketIO

# Import modularized components
from config import MODEL_PATH, ARDUINO_PORT
from controllers.arduino_controller import ArduinoController
from controllers.voice_controller import VoiceController
from controllers.robot_controller import RobotController
from controllers.manipulator_robot import ManipulatorRobot
from utils.csv_utils import init_csv_file
from routes.api_routes import register_api_routes
from routes.websocket_handlers import register_websocket_handlers

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
robot, controller, arduino_controller, voice_controller = None, None, None, None

# Register routes and handlers
register_api_routes(app)
register_websocket_handlers(socketio)

# Global system state tracking
system_initialized = False
system_status_data = {'status': 'initializing', 'message': 'System starting...'}
robot_status_data = {'connected_arms': []}

# System initialization and cleanup
def init_system(model_path=MODEL_PATH):
    """Enhanced system initialization with Arduino support"""
    global robot, controller, arduino_controller, voice_controller, system_initialized, system_status_data, robot_status_data
    try:
        logger.info("Initializing robot system...")
        system_status_data = {'status': 'initializing', 'message': 'Connecting to Arduino...'}
        socketio.emit('system_status', system_status_data)
        
        # Initialize Arduino controller
        arduino_controller = ArduinoController(ARDUINO_PORT)
        if arduino_controller.connect():
            logger.info("Arduino controller connected")
        else:
            logger.warning("Arduino controller not connected - continuing without it")
        
        system_status_data = {'status': 'initializing', 'message': 'Connecting to robot...'}
        socketio.emit('system_status', system_status_data)
        
        # Initialize robot
        robot = ManipulatorRobot()
        if not robot.connect():
            error_msg = 'Robot connection failed - no robot arms could be connected'
            logger.error(error_msg)
            system_status_data = {'status': 'error', 'message': error_msg}
            socketio.emit('system_status', system_status_data)
            return False

        # Initialize voice controller (optional component)
        try:
            system_status_data = {'status': 'initializing', 'message': 'Initializing voice controller...'}
            socketio.emit('system_status', system_status_data)
            voice_controller = VoiceController()
            logger.info("Voice controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice controller: {e}")
            voice_controller = None

        # Emit which arms are connected
        connected_arms = list(robot.connected_arms)
        robot_status_data = {'connected_arms': connected_arms}
        socketio.emit('robot_status', robot_status_data)
        logger.info(f"Connected robot arms: {connected_arms}")
        
        system_status_data = {'status': 'initializing', 'message': 'Setting up robot arms...'}
        socketio.emit('system_status', system_status_data)
        
        # Setup robot arms (only for connected arms)
        if robot.is_arm_connected('follower'):
            robot.disable_torque('follower')
        
        if robot.is_arm_connected('leader'):
            if not robot.setup_control('leader'):
                logger.warning('Leader arm setup failed, but continuing...')
        
        system_status_data = {'status': 'initializing', 'message': 'Starting cameras and controller...'}
        socketio.emit('system_status', system_status_data)
        
        # Initialize controller
        controller = RobotController(robot, model_path, 'follower')
        if not controller.start():
            robot.disconnect()
            if arduino_controller:
                arduino_controller.disconnect()
            error_msg = 'Controller start failed - check cameras'
            logger.error(error_msg)
            system_status_data = {'status': 'error', 'message': error_msg}
            socketio.emit('system_status', system_status_data)
            return False
        
        logger.info("System initialization complete!")
        status_message = f'System ready - Robot: {connected_arms}, Arduino: {"Connected" if arduino_controller.is_connected() else "Disconnected"}'
        system_status_data = {'status': 'ready', 'message': status_message}
        system_initialized = True
        
        # Broadcast updated status to ALL connected clients
        socketio.emit('system_status', system_status_data)
        socketio.emit('robot_status', robot_status_data)
        if arduino_controller:
            socketio.emit('arduino_status', arduino_controller.get_status())
        
        # Arduino startup effect
        if arduino_controller and arduino_controller.is_connected():
            arduino_controller.trigger_led_effect(3)  # Green fade for successful init
        
        if voice_controller:
            voice_controller.start()

        logger.info("Broadcasted system ready status to all connected clients")
        return True
        
    except Exception as e:
        error_msg = f'System initialization error: {str(e)}'
        logger.error(error_msg)
        system_status_data = {'status': 'error', 'message': error_msg}
        socketio.emit('system_status', system_status_data)
        return False

def cleanup_system():
    """Graceful system cleanup"""
    global system_initialized, system_status_data, robot_status_data
    logger.info("Cleaning up system...")
    system_initialized = False
    system_status_data = {'status': 'error', 'message': 'System shutting down'}
    robot_status_data = {'connected_arms': []}
    
    if controller:
        try: 
            controller.stop()
        except Exception as e:
            logger.error(f"Controller cleanup error: {e}")
    if robot:
        try: 
            robot.disconnect()
        except Exception as e:
            logger.error(f"Robot cleanup error: {e}")
    if arduino_controller:
        try:
            arduino_controller.disconnect()
        except Exception as e:
            logger.error(f"Arduino cleanup error: {e}")
    if voice_controller:
        try:
            voice_controller.stop()
        except Exception as e:
            logger.error(f"Voice controller cleanup error: {e}")
    logger.info("System cleanup complete")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info('Received shutdown signal, cleaning up...')
    cleanup_system()
    sys.exit(0)

def main():
    """Main application entry point"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize CSV file
    init_csv_file()
    
    # Start system initialization in background
    threading.Thread(target=init_system, daemon=True).start()
    
    try:
        logger.info("Starting Flask-SocketIO server on http://0.0.0.0:5000")
        logger.info("Press Ctrl+C to shutdown")
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()