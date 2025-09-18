#!/usr/bin/env python3
"""
Unified Follower Arm Controller
Handles mixed motor types: XL430-W250-T (3 motors) + XL330-M288-T (1 motor)
Complete terminal interface for position reading, torque control, and real-time monitoring
"""

import os
import time
import signal
import sys
import threading
import json
from dynamixel_sdk import *

# Load hardware configuration
def load_hardware_config():
    """Load hardware configuration from hardware_config.json"""
    config_file = 'hardware_config.json'
    default_config = {
        'port': '/dev/follower_arm',
        'baudrate': 1000000
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                follower_config = config.get('robot_arms', {}).get('follower', {})
                return {
                    'port': follower_config.get('port', default_config['port']),
                    'baudrate': follower_config.get('baudrate', default_config['baudrate'])
                }
        except Exception as e:
            print(f"Warning: Failed to load hardware config: {e}")
            print("Using default configuration")
    
    return default_config

# Load configuration
hw_config = load_hardware_config()

# Dynamixel Configuration
PROTOCOL_VERSION = 2.0
BAUDRATE = hw_config['baudrate']
DEVICE_NAME = hw_config['port']

# Motor Configuration - Mixed Types
MOTOR_CONFIGS = {
    # XL430-W250-T motors (IDs 1, 2, 3)
    1: {
        'model': 'XL430-W250-T',
        'model_number': 1060,
        'position_center': 2048,
        'position_min': 0,
        'position_max': 4095,
        'angle_resolution': 0.088  # degrees per unit
    },
    2: {
        'model': 'XL430-W250-T', 
        'model_number': 1060,
        'position_center': 2048,
        'position_min': 0,
        'position_max': 4095,
        'angle_resolution': 0.088
    },
    3: {
        'model': 'XL430-W250-T',
        'model_number': 1060,
        'position_center': 2048,
        'position_min': 0,
        'position_max': 4095,
        'angle_resolution': 0.088
    },
    # XL330-M288-T motor (ID 4)
    4: {
        'model': 'XL330-M288-T',
        'model_number': 1200,
        'position_center': 2048,
        'position_min': 0,
        'position_max': 4095,
        'angle_resolution': 0.088
    }
}

# Control Table Addresses (Protocol 2.0)
# These are the same for both XL430 and XL330 series
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_TEMPERATURE = 146

# Motor IDs
MOTOR_IDS = [1, 2, 3, 4]

# Global variables
port_handler = None
packet_handler = None
connected_motors = []
monitoring_active = False
monitoring_thread = None

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=Colors.ENDC):
    """Print text with color"""
    print(f"{color}{text}{Colors.ENDC}")

def initialize_dynamixel():
    """Initialize Dynamixel communication"""
    global port_handler, packet_handler
    
    print_colored("🔧 Initializing Dynamixel communication...", Colors.CYAN)
    
    port_handler = PortHandler(DEVICE_NAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    if not port_handler.openPort():
        print_colored(f"✗ Failed to open port {DEVICE_NAME}", Colors.FAIL)
        return False
    
    if not port_handler.setBaudRate(BAUDRATE):
        print_colored(f"✗ Failed to set baudrate to {BAUDRATE}", Colors.FAIL)
        return False
    
    print_colored(f"✓ Port {DEVICE_NAME} opened successfully", Colors.GREEN)
    print_colored(f"✓ Baudrate set to {BAUDRATE}", Colors.GREEN)
    
    return True

def ping_motors():
    """Ping all motors and return connected ones with type detection"""
    global connected_motors
    
    print_colored(f"\n📡 Scanning for follower arm motors {MOTOR_IDS}...", Colors.CYAN)
    
    connected_motors = []
    for motor_id in MOTOR_IDS:
        model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
        if comm_result == COMM_SUCCESS:
            expected_config = MOTOR_CONFIGS[motor_id]
            
            if model_number == expected_config['model_number']:
                print_colored(f"✓ Motor {motor_id}: {expected_config['model']} connected (Model: {model_number})", Colors.GREEN)
                connected_motors.append(motor_id)
            else:
                print_colored(f"⚠️  Motor {motor_id}: Wrong model (Expected: {expected_config['model_number']}, Got: {model_number})", Colors.WARNING)
        else:
            expected_config = MOTOR_CONFIGS[motor_id]
            print_colored(f"✗ Motor {motor_id}: {expected_config['model']} not found", Colors.WARNING)
    
    if connected_motors:
        print_colored(f"✅ Found {len(connected_motors)} motors: {connected_motors}", Colors.GREEN)
    else:
        print_colored("❌ No motors found!", Colors.FAIL)
    
    return len(connected_motors) > 0

def read_motor_status(motor_ids=None, show_header=True):
    """Read comprehensive status of specified motors"""
    if motor_ids is None:
        motor_ids = connected_motors
    
    if show_header:
        print_colored("\n📋 Follower Arm Motor Status:", Colors.HEADER)
        print_colored("-" * 110, Colors.HEADER)
    
    status_data = {}
    
    for motor_id in motor_ids:
        config = MOTOR_CONFIGS[motor_id]
        
        # Torque state
        torque_enable, comm_result, error = packet_handler.read1ByteTxRx(
            port_handler, motor_id, ADDR_TORQUE_ENABLE)
        torque_enabled = (comm_result == COMM_SUCCESS and error == 0 and torque_enable)
        torque_str = f"{Colors.GREEN}ENABLED {Colors.ENDC}" if torque_enabled else f"{Colors.WARNING}DISABLED{Colors.ENDC}"
        
        # Present Position
        position, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_POSITION)
        if comm_result == COMM_SUCCESS and error == 0:
            angle = (position - config['position_center']) * config['angle_resolution']
            pos_str = f"{position:4d} ({angle:+6.2f}°)"
        else:
            pos_str = "---- (  ----°)"
        
        # Goal Position
        goal_position, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_GOAL_POSITION)
        if comm_result == COMM_SUCCESS and error == 0:
            goal_angle = (goal_position - config['position_center']) * config['angle_resolution']
            goal_str = f"{goal_position:4d} ({goal_angle:+6.2f}°)"
            
            # Check if motor reached goal
            if position != -1:  # valid present position
                diff = abs(position - goal_position)
                if diff <= 10:
                    status_icon = f"{Colors.GREEN}✓{Colors.ENDC}"
                else:
                    status_icon = f"{Colors.WARNING}→{Colors.ENDC}"
            else:
                status_icon = "?"
        else:
            goal_str = "---- (  ----°)"
            status_icon = "?"
        
        # Velocity
        velocity, comm_result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_VELOCITY)
        if comm_result == COMM_SUCCESS and error == 0:
            # Different velocity conversion for different motors
            if config['model'] == 'XL430-W250-T':
                rpm = velocity * 0.229 if velocity < 2147483648 else (velocity - 4294967296) * 0.229
            else:  # XL330-M288-T
                rpm = velocity * 0.229 if velocity < 2147483648 else (velocity - 4294967296) * 0.229
            vel_str = f"{rpm:+6.1f} RPM"
        else:
            vel_str = "  ---- RPM"
        
        # Current
        current, comm_result, error = packet_handler.read2ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_CURRENT)
        if comm_result == COMM_SUCCESS and error == 0:
            # Different current conversion for different motors
            if config['model'] == 'XL430-W250-T':
                current_ma = current * 2.69 if current < 32768 else (current - 65536) * 2.69
            else:  # XL330-M288-T
                current_ma = current * 2.69 if current < 32768 else (current - 65536) * 2.69
            current_str = f"{current_ma:+6.1f} mA"
        else:
            current_str = "  ---- mA"
        
        # Motor type indicator
        type_str = f"{Colors.BLUE}{config['model'][:5]}{Colors.ENDC}"
        
        print(f"  Motor {motor_id} ({type_str}): Torque {torque_str} | Present {pos_str} | Goal {goal_str} {status_icon} | Vel {vel_str} | Current {current_str}")
        
        status_data[motor_id] = {
            'model': config['model'],
            'torque_enabled': torque_enabled,
            'position': position if comm_result == COMM_SUCCESS and error == 0 else None,
            'goal_position': goal_position if comm_result == COMM_SUCCESS and error == 0 else None,
            'angle': angle if comm_result == COMM_SUCCESS and error == 0 else None,
            'velocity': velocity if comm_result == COMM_SUCCESS and error == 0 else None,
            'current': current if comm_result == COMM_SUCCESS and error == 0 else None
        }
    
    return status_data

def validate_position(motor_id, position):
    """Validate if position is within safe range for specific motor"""
    if motor_id not in MOTOR_CONFIGS:
        return False, f"Unknown motor ID {motor_id}"
    
    if not isinstance(position, (int, float)):
        return False, "Position must be a number"
    
    config = MOTOR_CONFIGS[motor_id]
    position = int(position)
    
    if position < config['position_min'] or position > config['position_max']:
        return False, f"Position must be between {config['position_min']} and {config['position_max']} for {config['model']}"
    
    return True, position

def angle_to_position(motor_id, angle):
    """Convert angle in degrees to motor position for specific motor"""
    config = MOTOR_CONFIGS[motor_id]
    return int(config['position_center'] + (angle / config['angle_resolution']))

def position_to_angle(motor_id, position):
    """Convert motor position to angle in degrees for specific motor"""
    config = MOTOR_CONFIGS[motor_id]
    return (position - config['position_center']) * config['angle_resolution']

def set_torque_state(motor_id, enable):
    """Set torque state for a specific motor"""
    action = "Enabling" if enable else "Disabling"
    
    comm_result, error = packet_handler.write1ByteTxRx(
        port_handler, motor_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)
    
    if comm_result == COMM_SUCCESS and error == 0:
        status = f"{Colors.GREEN}✓{Colors.ENDC}"
        config = MOTOR_CONFIGS[motor_id]
        print(f"  Motor {motor_id} ({config['model'][:5]}): {action} torque... {status}")
        return True
    else:
        status = f"{Colors.FAIL}✗ {packet_handler.getTxRxResult(comm_result)}{Colors.ENDC}"
        print(f"  Motor {motor_id}: {action} torque... {status}")
        return False

def set_all_torque_states(enable):
    """Set torque state for all connected motors"""
    action = "Enabling" if enable else "Disabling"
    print_colored(f"\n⚡ {action} torque for all follower arm motors...", Colors.CYAN)
    
    success_count = 0
    for motor_id in connected_motors:
        if set_torque_state(motor_id, enable):
            success_count += 1
    
    if success_count == len(connected_motors):
        print_colored(f"✅ All {success_count} motors torque {'enabled' if enable else 'disabled'}", Colors.GREEN)
    else:
        print_colored(f"⚠️  {success_count}/{len(connected_motors)} motors torque {'enabled' if enable else 'disabled'}", Colors.WARNING)
    
    return success_count == len(connected_motors)

def set_goal_position(motor_id, position):
    """Set goal position for a specific motor"""
    # Validate position
    valid, result = validate_position(motor_id, position)
    if not valid:
        print_colored(f"❌ Invalid position: {result}", Colors.FAIL)
        return False
    
    position = result
    angle = position_to_angle(motor_id, position)
    config = MOTOR_CONFIGS[motor_id]
    
    print(f"  Motor {motor_id} ({config['model'][:5]}): Setting goal position {position} ({angle:+6.2f}°)...", end=" ")
    
    comm_result, error = packet_handler.write4ByteTxRx(
        port_handler, motor_id, ADDR_GOAL_POSITION, position)
    
    if comm_result == COMM_SUCCESS and error == 0:
        print_colored("✓", Colors.GREEN)
        return True
    else:
        print_colored(f"✗ {packet_handler.getTxRxResult(comm_result)}", Colors.FAIL)
        return False

def move_motor_to_angle(motor_id, angle):
    """Move motor to specific angle in degrees"""
    position = angle_to_position(motor_id, angle)
    
    # Validate converted position
    valid, validated_pos = validate_position(motor_id, position)
    if not valid:
        print_colored(f"❌ Angle {angle}° converts to invalid position: {validated_pos}", Colors.FAIL)
        return False
    
    config = MOTOR_CONFIGS[motor_id]
    print_colored(f"\n🎯 Moving Motor {motor_id} ({config['model']}) to {angle:+6.2f}° (position {validated_pos})...", Colors.CYAN)
    return set_goal_position(motor_id, validated_pos)

def move_to_center_position():
    """Move all motors to center position (0 degrees)"""
    print_colored(f"\n🎯 Moving all follower arm motors to center position (0°)...", Colors.CYAN)
    
    success_count = 0
    for motor_id in connected_motors:
        config = MOTOR_CONFIGS[motor_id]
        if set_goal_position(motor_id, config['position_center']):
            success_count += 1
    
    if success_count == len(connected_motors):
        print_colored(f"✅ All {success_count} motors moved to center", Colors.GREEN)
    else:
        print_colored(f"⚠️  {success_count}/{len(connected_motors)} motors moved successfully", Colors.WARNING)
    
    return success_count == len(connected_motors)

def wait_for_movement_complete(motor_ids=None, timeout=10, threshold=10):
    """Wait for motors to reach their goal positions"""
    if motor_ids is None:
        motor_ids = connected_motors
    
    print_colored(f"⏳ Waiting for movement to complete...", Colors.CYAN)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        all_reached = True
        
        for motor_id in motor_ids:
            # Read current position
            current_pos, comm_result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION)
            
            if comm_result == COMM_SUCCESS and error == 0:
                # Read goal position
                goal_pos, comm_result, error = packet_handler.read4ByteTxRx(
                    port_handler, motor_id, ADDR_GOAL_POSITION)
                
                if comm_result == COMM_SUCCESS and error == 0:
                    diff = abs(current_pos - goal_pos)
                    if diff > threshold:
                        all_reached = False
                        break
        
        if all_reached:
            elapsed = time.time() - start_time
            print_colored(f"✅ Movement completed in {elapsed:.1f}s", Colors.GREEN)
            return True
        
        time.sleep(0.1)
    
    print_colored(f"⚠️  Movement timeout after {timeout}s", Colors.WARNING)
    return False

def real_time_monitor():
    """Real-time position monitoring in background thread"""
    global monitoring_active
    
    last_positions = {}
    update_count = 0
    
    print_colored("🔄 Real-time monitoring started (Press 'm' to stop)", Colors.CYAN)
    print_colored("📍 Move the follower arm to see position changes", Colors.CYAN)
    print_colored("-" * 90, Colors.HEADER)
    
    while monitoring_active:
        current_positions = {}
        position_changed = False
        
        # Read all motor positions
        for motor_id in connected_motors:
            position, comm_result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION)
            
            if comm_result == COMM_SUCCESS and error == 0:
                config = MOTOR_CONFIGS[motor_id]
                angle = (position - config['position_center']) * config['angle_resolution']
                current_positions[motor_id] = {
                    'raw': position,
                    'angle': angle
                }
                
                # Check if position changed significantly (>3 units)
                if (motor_id not in last_positions or 
                    abs(position - last_positions[motor_id]['raw']) > 3):
                    position_changed = True
        
        # Only print if positions changed or every 20 updates
        if position_changed or update_count % 20 == 0:
            timestamp = time.strftime('%H:%M:%S')
            print(f"\r{Colors.BLUE}⏰ {timestamp}{Colors.ENDC} | ", end="")
            
            for motor_id in connected_motors:
                config = MOTOR_CONFIGS[motor_id]
                type_abbr = "430" if "XL430" in config['model'] else "330"
                
                if motor_id in current_positions:
                    pos_data = current_positions[motor_id]
                    print(f"M{motor_id}({type_abbr}): {pos_data['raw']:4d} ({pos_data['angle']:+6.2f}°) | ", end="")
                else:
                    print(f"M{motor_id}({type_abbr}): ---- (  ----°) | ", end="")
            print("")
        
        last_positions = current_positions.copy()
        update_count += 1
        
        time.sleep(0.05)  # 20Hz update rate

def start_monitoring():
    """Start real-time monitoring in background thread"""
    global monitoring_active, monitoring_thread
    
    if monitoring_active:
        print_colored("⚠️  Monitoring is already active", Colors.WARNING)
        return
    
    monitoring_active = True
    monitoring_thread = threading.Thread(target=real_time_monitor, daemon=True)
    monitoring_thread.start()

def stop_monitoring():
    """Stop real-time monitoring"""
    global monitoring_active
    
    if not monitoring_active:
        print_colored("⚠️  Monitoring is not active", Colors.WARNING)
        return
    
    monitoring_active = False
    print_colored("\n🛑 Real-time monitoring stopped", Colors.WARNING)

def show_help():
    """Show available commands"""
    print_colored("\n🎮 Follower Arm Controller - Available Commands:", Colors.HEADER)
    print_colored("=" * 80, Colors.HEADER)
    print_colored("📊 Status & Monitoring:", Colors.CYAN)
    print("  's' or 'status'     - Show current motor status")
    print("  'm' or 'monitor'    - Toggle real-time position monitoring")
    print("  'p' or 'ping'       - Re-scan for connected motors")
    
    print_colored("\n⚡ Torque Control:", Colors.CYAN)
    print("  'e' or 'enable'     - Enable torque for all motors")
    print("  'd' or 'disable'    - Disable torque for all motors")
    print("  'e1', 'e2', etc     - Enable torque for specific motor")
    print("  'd1', 'd2', etc     - Disable torque for specific motor")
    
    print_colored("\n🎯 Position Control:", Colors.CYAN)
    print("  'center'             - Move all motors to center position (0°)")
    print("  'move M A'           - Move motor M to angle A degrees")
    print("                         Example: 'move 1 45' moves motor 1 to 45°")
    print("  'pos M P'            - Move motor M to position P (0-4095)")
    print("                         Example: 'pos 2 2500' moves motor 2 to position 2500")
    print("  'wait'               - Wait for current movement to complete")
    
    print_colored("\n🔧 System:", Colors.CYAN)
    print("  'h' or 'help'       - Show this help")
    print("  'c' or 'clear'      - Clear screen")
    print("  'q' or 'quit'       - Quit controller")
    print_colored("=" * 80, Colors.HEADER)
    print_colored("💡 Motor Types:", Colors.WARNING)
    print("  - Motors 1-3: XL430-W250-T")
    print("  - Motor 4:    XL330-M288-T")
    print("  - All motors: Center=2048 (0°), Range=0-4095 (≈±180°)")
    print("  - Motors must have torque enabled to move")

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def main():
    global monitoring_active
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        global monitoring_active
        monitoring_active = False
        print_colored('\n\n🛑 Shutting down...', Colors.WARNING)
        cleanup_and_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Header
    clear_screen()
    print_colored("🦾 Follower Arm Controller (Mixed Motors)", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 70, Colors.HEADER)
    print(f"Device: {DEVICE_NAME}")
    print(f"Baudrate: {BAUDRATE}")
    print(f"Expected Motors: {MOTOR_IDS}")
    print(f"Motor Types: XL430-W250-T (1,2,3) + XL330-M288-T (4)")
    print_colored("=" * 70, Colors.HEADER)
    
    # Initialize
    if not initialize_dynamixel():
        print_colored("❌ Failed to initialize. Exiting.", Colors.FAIL)
        return
    
    if not ping_motors():
        print_colored("❌ No motors found. Exiting.", Colors.FAIL)
        return
    
    # Initial status
    read_motor_status()
    
    # Show help
    show_help()
    
    # Main command loop
    while True:
        try:
            if monitoring_active:
                cmd = input(f"\n{Colors.CYAN}🎯 [MONITORING] Enter command (or 'm' to stop monitoring): {Colors.ENDC}").strip().lower()
            else:
                cmd = input(f"\n{Colors.CYAN}🎯 Enter command: {Colors.ENDC}").strip().lower()
            
            if cmd in ['q', 'quit', 'exit']:
                break
                
            elif cmd in ['h', 'help']:
                show_help()
                
            elif cmd in ['c', 'clear']:
                clear_screen()
                print_colored("🦾 Follower Arm Controller", Colors.HEADER + Colors.BOLD)
                
            elif cmd in ['s', 'status']:
                read_motor_status()
                
            elif cmd in ['p', 'ping']:
                ping_motors()
                
            elif cmd in ['m', 'monitor']:
                if monitoring_active:
                    stop_monitoring()
                else:
                    start_monitoring()
                    
            elif cmd in ['e', 'enable']:
                set_all_torque_states(True)
                time.sleep(0.5)
                if not monitoring_active:
                    read_motor_status()
                    
            elif cmd in ['d', 'disable']:
                set_all_torque_states(False)
                time.sleep(0.5)
                if not monitoring_active:
                    read_motor_status()
                    
            elif cmd.startswith('e') and len(cmd) == 2 and cmd[1].isdigit():
                motor_id = int(cmd[1])
                if motor_id in connected_motors:
                    set_torque_state(motor_id, True)
                    time.sleep(0.5)
                    if not monitoring_active:
                        read_motor_status([motor_id])
                else:
                    print_colored(f"❌ Motor {motor_id} not connected", Colors.FAIL)
                    
            elif cmd.startswith('d') and len(cmd) == 2 and cmd[1].isdigit():
                motor_id = int(cmd[1])
                if motor_id in connected_motors:
                    set_torque_state(motor_id, False)
                    time.sleep(0.5)
                    if not monitoring_active:
                        read_motor_status([motor_id])
                else:
                    print_colored(f"❌ Motor {motor_id} not connected", Colors.FAIL)
            
            # Position Control Commands
            elif cmd == 'center':
                # Check if any motor has torque enabled
                torque_enabled = False
                for motor_id in connected_motors:
                    torque_enable, comm_result, error = packet_handler.read1ByteTxRx(
                        port_handler, motor_id, ADDR_TORQUE_ENABLE)
                    if comm_result == COMM_SUCCESS and error == 0 and torque_enable:
                        torque_enabled = True
                        break
                
                if not torque_enabled:
                    print_colored("⚠️  Warning: No motors have torque enabled. Enable torque first.", Colors.WARNING)
                else:
                    move_to_center_position()
                    if not monitoring_active:
                        time.sleep(1)
                        read_motor_status()
                        
            elif cmd.startswith('move '):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        motor_id = int(parts[1])
                        angle = float(parts[2])
                        
                        if motor_id not in connected_motors:
                            print_colored(f"❌ Motor {motor_id} not connected", Colors.FAIL)
                        else:
                            # Check if motor has torque enabled
                            torque_enable, comm_result, error = packet_handler.read1ByteTxRx(
                                port_handler, motor_id, ADDR_TORQUE_ENABLE)
                            
                            if comm_result == COMM_SUCCESS and error == 0 and torque_enable:
                                move_motor_to_angle(motor_id, angle)
                                if not monitoring_active:
                                    time.sleep(1)
                                    read_motor_status([motor_id])
                            else:
                                print_colored(f"⚠️  Motor {motor_id} torque is disabled. Enable torque first.", Colors.WARNING)
                                
                    except ValueError:
                        print_colored("❌ Invalid format. Use: move MOTOR_ID ANGLE", Colors.FAIL)
                        print_colored("   Example: move 1 45", Colors.FAIL)
                else:
                    print_colored("❌ Invalid format. Use: move MOTOR_ID ANGLE", Colors.FAIL)
                    print_colored("   Example: move 1 45", Colors.FAIL)
                    
            elif cmd.startswith('pos '):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        motor_id = int(parts[1])
                        position = int(parts[2])
                        
                        if motor_id not in connected_motors:
                            print_colored(f"❌ Motor {motor_id} not connected", Colors.FAIL)
                        else:
                            # Check if motor has torque enabled
                            torque_enable, comm_result, error = packet_handler.read1ByteTxRx(
                                port_handler, motor_id, ADDR_TORQUE_ENABLE)
                            
                            if comm_result == COMM_SUCCESS and error == 0 and torque_enable:
                                angle = position_to_angle(motor_id, position)
                                config = MOTOR_CONFIGS[motor_id]
                                print_colored(f"\n🎯 Moving Motor {motor_id} ({config['model']}) to position {position} ({angle:+6.2f}°)...", Colors.CYAN)
                                set_goal_position(motor_id, position)
                                if not monitoring_active:
                                    time.sleep(1)
                                    read_motor_status([motor_id])
                            else:
                                print_colored(f"⚠️  Motor {motor_id} torque is disabled. Enable torque first.", Colors.WARNING)
                                
                    except ValueError:
                        print_colored("❌ Invalid format. Use: pos MOTOR_ID POSITION", Colors.FAIL)
                        print_colored("   Example: pos 2 2500", Colors.FAIL)
                else:
                    print_colored("❌ Invalid format. Use: pos MOTOR_ID POSITION", Colors.FAIL)
                    print_colored("   Example: pos 2 2500", Colors.FAIL)
                    
            elif cmd == 'wait':
                wait_for_movement_complete()
                if not monitoring_active:
                    read_motor_status()
                    
            elif cmd == '':
                continue  # Empty command, just continue
                
            else:
                print_colored("❓ Unknown command. Type 'h' for help.", Colors.WARNING)
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as e:
            print_colored(f"❌ Error: {e}", Colors.FAIL)
    
    cleanup_and_exit()

def cleanup_and_exit():
    """Cleanup before exit"""
    global monitoring_active, port_handler
    
    # Stop monitoring
    if monitoring_active:
        monitoring_active = False
        print_colored("\n🛑 Stopping monitoring...", Colors.WARNING)
        time.sleep(0.2)
    
    # Disable all torques
    if port_handler and connected_motors:
        print_colored("🧹 Disabling all torques...", Colors.WARNING)
        set_all_torque_states(False)
    
    # Close port
    if port_handler:
        port_handler.closePort()
        print_colored("🔌 Port closed", Colors.GREEN)
    
    print_colored("👋 Follower Arm Controller stopped", Colors.GREEN)

if __name__ == "__main__":
    main()