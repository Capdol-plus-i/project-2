#!/usr/bin/env python3
"""
Leader-Follower Arm Synchronization Controller
Real-time synchronization between leader arm (XL330-M077-T x4) and follower arm (XL430-W250-T x3 + XL330-M288-T x1)
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
        'leader': {'port': '/dev/leader_arm', 'baudrate': 1000000},
        'follower': {'port': '/dev/ttyACM2', 'baudrate': 1000000}
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                robot_config = config.get('robot_arms', {})
                return {
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
            print("Using default configuration")
    
    return default_config

# Load configuration
hw_config = load_hardware_config()

# Dynamixel Configuration
PROTOCOL_VERSION = 2.0

# Control Table Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126

# Motor Configuration
MOTOR_IDS = [1, 2, 3, 4]

# Leader Arm Configuration (XL330-M077-T x4)
LEADER_CONFIG = {
    'port': hw_config['leader']['port'],
    'baudrate': hw_config['leader']['baudrate'],
    'motors': {
        1: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088},
        2: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088},
        3: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088},
        4: {'model': 'XL330-M077-T', 'center': 2048, 'resolution': 0.088}
    }
}

# Follower Arm Configuration (XL430-W250-T x3 + XL330-M288-T x1)
FOLLOWER_CONFIG = {
    'port': hw_config['follower']['port'],
    'baudrate': hw_config['follower']['baudrate'],
    'motors': {
        1: {'model': 'XL430-W250-T', 'center': 2048, 'resolution': 0.088},
        2: {'model': 'XL430-W250-T', 'center': 2048, 'resolution': 0.088},
        3: {'model': 'XL430-W250-T', 'center': 2048, 'resolution': 0.088},
        4: {'model': 'XL330-M288-T', 'center': 2048, 'resolution': 0.088}
    }
}

# Global variables
leader_port_handler = None
leader_packet_handler = None
follower_port_handler = None
follower_packet_handler = None
connected_leader_motors = []
connected_follower_motors = []
sync_active = False
sync_thread = None

# Position offset for calibration (angle differences between leader and follower)
position_offsets = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}  # degrees
# Per-motor rotation direction mapping: +1 (same) or -1 (inverted)
direction_multipliers = {1: 1, 2: 1, 3: 1, 4: 1}
# Leader→Follower motor ID map (default identity)
id_map = {1: 1, 2: 2, 3: 3, 4: 4}

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

def initialize_dynamixel(config, name):
    """Initialize Dynamixel communication for leader or follower"""
    print_colored(f"🔧 Initializing {name} arm communication...", Colors.CYAN)
    
    port_handler = PortHandler(config['port'])
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    if not port_handler.openPort():
        print_colored(f"✗ Failed to open {name} port {config['port']}", Colors.FAIL)
        return None, None
    
    if not port_handler.setBaudRate(config['baudrate']):
        print_colored(f"✗ Failed to set {name} baudrate to {config['baudrate']}", Colors.FAIL)
        return None, None
    
    print_colored(f"✓ {name} arm connected: {config['port']} @ {config['baudrate']}", Colors.GREEN)
    
    return port_handler, packet_handler

def ping_motors(port_handler, packet_handler, config, name):
    """Ping all motors and return connected ones"""
    print_colored(f"\n📡 Scanning {name} arm motors...", Colors.CYAN)
    
    connected_motors = []
    for motor_id in MOTOR_IDS:
        model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
        if comm_result == COMM_SUCCESS:
            motor_config = config['motors'][motor_id]
            print_colored(f"✓ {name} Motor {motor_id}: {motor_config['model']} (Model: {model_number})", Colors.GREEN)
            connected_motors.append(motor_id)
        else:
            motor_config = config['motors'][motor_id]
            print_colored(f"✗ {name} Motor {motor_id}: {motor_config['model']} not found", Colors.WARNING)
    
    return connected_motors

def read_motor_position(port_handler, packet_handler, motor_id, config):
    """Read current position from a specific motor"""
    position, comm_result, error = packet_handler.read4ByteTxRx(
        port_handler, motor_id, ADDR_PRESENT_POSITION)
    
    if comm_result == COMM_SUCCESS and error == 0:
        motor_config = config['motors'][motor_id]
        angle = (position - motor_config['center']) * motor_config['resolution']
        return position, angle, True
    else:
        return None, None, False

def set_motor_position(port_handler, packet_handler, motor_id, position, config):
    """Set goal position for a specific motor"""
    # Validate position range (0-4095 for both motor types)
    position = max(0, min(4095, int(position)))
    
    comm_result, error = packet_handler.write4ByteTxRx(
        port_handler, motor_id, ADDR_GOAL_POSITION, position)
    
    return comm_result == COMM_SUCCESS and error == 0

def set_torque_state(port_handler, packet_handler, motor_id, enable):
    """Set torque state for a specific motor"""
    comm_result, error = packet_handler.write1ByteTxRx(
        port_handler, motor_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)
    
    return comm_result == COMM_SUCCESS and error == 0

def enable_follower_torques():
    """Enable torque for all connected follower motors"""
    print_colored("\n⚡ Enabling follower arm torques...", Colors.CYAN)
    
    success_count = 0
    for motor_id in connected_follower_motors:
        if set_torque_state(follower_port_handler, follower_packet_handler, motor_id, True):
            print_colored(f"  ✓ Follower Motor {motor_id} torque enabled", Colors.GREEN)
            success_count += 1
        else:
            print_colored(f"  ✗ Follower Motor {motor_id} torque enable failed", Colors.FAIL)
    
    return success_count == len(connected_follower_motors)

def disable_follower_torques():
    """Disable torque for all connected follower motors"""
    print_colored("\n⚡ Disabling follower arm torques...", Colors.WARNING)
    
    for motor_id in connected_follower_motors:
        if set_torque_state(follower_port_handler, follower_packet_handler, motor_id, False):
            print_colored(f"  ✓ Follower Motor {motor_id} torque disabled", Colors.GREEN)
        else:
            print_colored(f"  ✗ Follower Motor {motor_id} torque disable failed", Colors.FAIL)

def angle_to_position(angle, motor_id, config):
    """Convert angle to position for specific motor"""
    motor_config = config['motors'][motor_id]
    return int(motor_config['center'] + (angle / motor_config['resolution']))

def sync_leader_to_follower():
    """Real-time synchronization thread - copies leader positions to follower"""
    global sync_active
    
    print_colored("🔄 Real-time synchronization started", Colors.CYAN)
    print_colored("📍 Move the leader arm to see follower arm follow", Colors.CYAN)
    print_colored("🛑 Press 'q' in main menu to stop synchronization", Colors.CYAN)
    print_colored("-" * 80, Colors.HEADER)
    
    last_positions = {}
    update_count = 0
    error_count = 0
    max_errors = 10  # Stop sync after too many consecutive errors
    
    while sync_active:
        try:
            positions_changed = False
            current_leader_positions = {}
            
            # Read all leader motor positions
            for leader_id in connected_leader_motors:
                follower_id = id_map.get(leader_id)
                if follower_id not in connected_follower_motors:
                    continue  # Skip if mapped follower motor not connected
                
                position, angle, success = read_motor_position(
                    leader_port_handler, leader_packet_handler, leader_id, LEADER_CONFIG)
                
                if success:
                    current_leader_positions[leader_id] = {'position': position, 'angle': angle}
                    
                    # Check if position changed significantly (>2 units)
                    if (leader_id not in last_positions or 
                        abs(position - last_positions[leader_id]['position']) > 2):
                        positions_changed = True
                else:
                    error_count += 1
                    if error_count > max_errors:
                        print_colored(f"\n⚠️  Too many read errors ({error_count}), stopping sync", Colors.WARNING)
                        sync_active = False
                        break
                    continue
            
            # If leader positions changed, update follower positions
            if positions_changed and current_leader_positions:
                success_count = 0
                
                for leader_id, leader_data in current_leader_positions.items():
                    follower_id = id_map.get(leader_id)
                    if follower_id in connected_follower_motors:
                        # Apply calibration offset & mapping
                        d = direction_multipliers.get(follower_id, 1)
                        offset = position_offsets.get(follower_id, 0.0)
                        target_angle = d * leader_data['angle'] + offset
                        target_position = angle_to_position(target_angle, follower_id, FOLLOWER_CONFIG)
                        
                        # Set follower motor position
                        if set_motor_position(follower_port_handler, follower_packet_handler, 
                                            follower_id, target_position, FOLLOWER_CONFIG):
                            success_count += 1
                
                # Display status every 10 updates or when positions change significantly
                if positions_changed or update_count % 20 == 0:
                    timestamp = time.strftime('%H:%M:%S')
                    print(f"\r{Colors.BLUE}⏰ {timestamp}{Colors.ENDC} | Syncing {success_count}/{len(current_leader_positions)} motors | ", end="")
                    
                    for leader_id in MOTOR_IDS:
                        if leader_id in current_leader_positions:
                            data = current_leader_positions[leader_id]
                            follower_id = id_map.get(leader_id)
                            d = direction_multipliers.get(follower_id, 1)
                            off = position_offsets.get(follower_id, 0.0)
                            target_angle = d * data['angle'] + off
                            print(f"L{leader_id}→F{follower_id}: {data['angle']:+6.1f}°→{target_angle:+6.1f}° (dir {d:+d}) | ", end="")
                        else:
                            print(f"L{leader_id}: ---- | ", end="")
                    print("", end="", flush=True)
                
                # Reset error count on successful sync
                error_count = 0
            
            last_positions = current_leader_positions.copy()
            update_count += 1
            
            # Sleep for smooth operation (20Hz)
            time.sleep(0.05)
            
        except Exception as e:
            print_colored(f"\n❌ Sync error: {e}", Colors.FAIL)
            error_count += 1
            if error_count > max_errors:
                sync_active = False
                break
            time.sleep(0.1)
    
    print_colored("\n🛑 Synchronization stopped", Colors.WARNING)

def start_synchronization():
    """Start leader-follower synchronization"""
    global sync_active, sync_thread
    
    if sync_active:
        print_colored("⚠️  Synchronization is already running", Colors.WARNING)
        return False
    
    # Enable follower torques
    if not enable_follower_torques():
        print_colored("❌ Failed to enable follower torques", Colors.FAIL)
        return False
    
    # Start sync thread
    sync_active = True
    sync_thread = threading.Thread(target=sync_leader_to_follower, daemon=True)
    sync_thread.start()
    
    return True

def stop_synchronization():
    """Stop leader-follower synchronization"""
    global sync_active
    
    if not sync_active:
        print_colored("⚠️  Synchronization is not running", Colors.WARNING)
        return
    
    sync_active = False
    print_colored("⏳ Stopping synchronization...", Colors.WARNING)
    time.sleep(0.2)
    
    # Disable follower torques
    disable_follower_torques()

def calibrate_position(motor_id, offset_angle):
    """Set position offset for a specific motor"""
    if motor_id in MOTOR_IDS:
        position_offsets[motor_id] = offset_angle
        print_colored(f"✓ Motor {motor_id} offset set to {offset_angle:+.2f}°", Colors.GREEN)
        return True
    else:
        print_colored(f"❌ Invalid motor ID: {motor_id}", Colors.FAIL)
        return False

def move_to_zero_position():
    """Move both leader and follower arms to zero position (center)"""
    print_colored("\n🎯 Moving both arms to zero position for calibration...", Colors.CYAN)
    
    # Move leader arm to center position
    print_colored("📍 Moving leader arm to center...", Colors.BLUE)
    leader_success = 0
    for motor_id in connected_leader_motors:
        center_pos = LEADER_CONFIG['motors'][motor_id]['center']
        if set_motor_position(leader_port_handler, leader_packet_handler, motor_id, center_pos, LEADER_CONFIG):
            print_colored(f"  ✓ Leader Motor {motor_id} → center position", Colors.GREEN)
            leader_success += 1
        else:
            print_colored(f"  ✗ Leader Motor {motor_id} move failed", Colors.FAIL)
    
    # Enable follower torques first
    print_colored("📍 Enabling follower torques...", Colors.BLUE)
    if not enable_follower_torques():
        print_colored("❌ Failed to enable follower torques", Colors.FAIL)
        return False
    
    # Move follower arm to center position
    print_colored("📍 Moving follower arm to center...", Colors.BLUE)
    follower_success = 0
    for motor_id in connected_follower_motors:
        center_pos = FOLLOWER_CONFIG['motors'][motor_id]['center']
        if set_motor_position(follower_port_handler, follower_packet_handler, motor_id, center_pos, FOLLOWER_CONFIG):
            print_colored(f"  ✓ Follower Motor {motor_id} → center position", Colors.GREEN)
            follower_success += 1
        else:
            print_colored(f"  ✗ Follower Motor {motor_id} move failed", Colors.FAIL)
    
    if leader_success > 0 and follower_success > 0:
        print_colored("✅ Both arms moved to zero position", Colors.GREEN)
        print_colored("⏳ Waiting 3 seconds for movement to complete...", Colors.WARNING)
        time.sleep(3)
        return True
    else:
        print_colored("❌ Failed to move arms to zero position", Colors.FAIL)
        return False

def auto_calibrate():
    """Automatically calculate and set calibration offsets"""
    print_colored("\n🔧 Starting automatic calibration...", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    
    # Step 1: Move to zero position
    if not move_to_zero_position():
        return False
    
    # Step 2: Read current positions
    print_colored("📊 Reading current positions for calibration...", Colors.CYAN)
    
    leader_positions = {}
    follower_positions = {}
    
    # Read leader positions
    for motor_id in connected_leader_motors:
        position, angle, success = read_motor_position(
            leader_port_handler, leader_packet_handler, motor_id, LEADER_CONFIG)
        if success:
            leader_positions[motor_id] = angle
            print_colored(f"  Leader Motor {motor_id}: {angle:+6.2f}°", Colors.BLUE)
        else:
            print_colored(f"  Leader Motor {motor_id}: Read failed", Colors.FAIL)
    
    # Read follower positions  
    for motor_id in connected_follower_motors:
        position, angle, success = read_motor_position(
            follower_port_handler, follower_packet_handler, motor_id, FOLLOWER_CONFIG)
        if success:
            follower_positions[motor_id] = angle
            print_colored(f"  Follower Motor {motor_id}: {angle:+6.2f}°", Colors.GREEN)
        else:
            print_colored(f"  Follower Motor {motor_id}: Read failed", Colors.FAIL)
    
    # Step 3: Calculate offsets
    print_colored("\n🧮 Calculating calibration offsets...", Colors.CYAN)
    calibrated_count = 0
    
    for leader_id in MOTOR_IDS:
        follower_id = id_map.get(leader_id)
        if leader_id in leader_positions and follower_id in follower_positions:
            # Model: follower_target = dir*leader_angle + offset
            # → offset[F] = follower[F] - dir[F]*leader[L]
            d = direction_multipliers.get(follower_id, 1)
            offset = follower_positions[follower_id] - d * leader_positions[leader_id]
            position_offsets[follower_id] = offset
            print_colored(f"  L{leader_id}→F{follower_id}: F {follower_positions[follower_id]:+6.2f}° - (dir {d:+d} * L {leader_positions[leader_id]:+6.2f}°) = {offset:+6.2f}°", Colors.CYAN)
            calibrated_count += 1
        elif leader_id in connected_leader_motors or follower_id in connected_follower_motors:
            print_colored(f"  L{leader_id}→F{follower_id}: Skipped (not available on both arms)", Colors.WARNING)
    
    # Step 4: Apply calibration test
    if calibrated_count > 0:
        print_colored(f"\n✅ Calibration complete! {calibrated_count} motors calibrated", Colors.GREEN)
        print_colored("🧪 Testing calibration by moving follower to match leader...", Colors.CYAN)
        
        test_success = 0
        for leader_id in connected_leader_motors:
            follower_id = id_map.get(leader_id)
            if follower_id in connected_follower_motors and leader_id in leader_positions:
                # Apply offset and move follower
                d = direction_multipliers.get(follower_id, 1)
                target_angle = d * leader_positions[leader_id] + position_offsets.get(follower_id, 0.0)
                target_position = angle_to_position(target_angle, follower_id, FOLLOWER_CONFIG)
                
                if set_motor_position(follower_port_handler, follower_packet_handler, follower_id, target_position, FOLLOWER_CONFIG):
                    print_colored(f"  ✓ Follower Motor {follower_id} moved to {target_angle:+6.2f}°", Colors.GREEN)
                    test_success += 1
        
        print_colored("⏳ Waiting 2 seconds to settle...", Colors.WARNING)
        time.sleep(2)
        
        # Verify calibration
        print_colored("\n🔍 Verifying calibration results...", Colors.CYAN)
        for leader_id in connected_leader_motors:
            follower_id = id_map.get(leader_id)
            if follower_id in connected_follower_motors and leader_id in leader_positions:
                position, angle, success = read_motor_position(
                    follower_port_handler, follower_packet_handler, follower_id, FOLLOWER_CONFIG)
                if success:
                    d = direction_multipliers.get(follower_id, 1)
                    target_angle = d * leader_positions[leader_id]
                    error = abs(angle - target_angle)
                    if error < 2.0:  # Within 2 degrees
                        print_colored(f"  ✓ Motor {follower_id}: {angle:+6.2f}° (target: {target_angle:+6.2f}°, error: {error:.2f}°)", Colors.GREEN)
                    else:
                        print_colored(f"  ⚠️  Motor {follower_id}: {angle:+6.2f}° (target: {target_angle:+6.2f}°, error: {error:.2f}°)", Colors.WARNING)
        
        print_colored("\n🎉 Auto-calibration completed!", Colors.GREEN)
        return True
    else:
        print_colored("\n❌ No motors available for calibration", Colors.FAIL)
        return False

def save_calibration(filename="calibration.json"):
    """Save current calibration offsets to file"""
    try:
        calibration_data = {
            'timestamp': time.time(),
            'position_offsets': position_offsets.copy(),
            'direction_multipliers': direction_multipliers.copy(),
            'id_map': id_map.copy(),
            'leader_config': LEADER_CONFIG,
            'follower_config': FOLLOWER_CONFIG
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print_colored(f"💾 Calibration saved to {filename}", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ Failed to save calibration: {e}", Colors.FAIL)
        return False

def load_calibration(filename="calibration.json"):
    """Load calibration offsets from file"""
    global position_offsets, direction_multipliers, id_map
    
    try:
        if not os.path.exists(filename):
            print_colored(f"❌ Calibration file {filename} not found", Colors.FAIL)
            return False
        
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
        
        position_offsets = calibration_data.get('position_offsets', {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
        direction_multipliers = calibration_data.get('direction_multipliers', {1: 1, 2: 1, 3: 1, 4: 1})
        id_map = calibration_data.get('id_map', {1: 1, 2: 2, 3: 3, 4: 4})
        
        # Convert string keys to int if needed
        if isinstance(list(position_offsets.keys())[0], str):
            position_offsets = {int(k): v for k, v in position_offsets.items()}
        if isinstance(list(direction_multipliers.keys())[0], str):
            direction_multipliers = {int(k): int(v) for k, v in direction_multipliers.items()}
        if isinstance(list(id_map.keys())[0], str):
            id_map = {int(k): int(v) for k, v in id_map.items()}
        
        print_colored(f"📁 Calibration loaded from {filename}", Colors.GREEN)
        for leader_id in MOTOR_IDS:
            follower_id = id_map.get(leader_id)
            off = position_offsets.get(follower_id, 0.0)
            d = direction_multipliers.get(follower_id, 1)
            print_colored(f"  L{leader_id}→F{follower_id}: offset {off:+6.2f}° | dir {d:+d}", Colors.CYAN)
        
        return True
    except Exception as e:
        print_colored(f"❌ Failed to load calibration: {e}", Colors.FAIL)
        return False

def show_status():
    """Show current status of both arms"""
    print_colored("\n📋 Leader-Follower Status:", Colors.HEADER)
    print_colored("-" * 80, Colors.HEADER)
    
    # Leader arm status
    print_colored("🎯 Leader Arm:", Colors.CYAN)
    for motor_id in connected_leader_motors:
        position, angle, success = read_motor_position(
            leader_port_handler, leader_packet_handler, motor_id, LEADER_CONFIG)
        if success:
            print(f"  Motor {motor_id}: {position:4d} ({angle:+6.2f}°)")
        else:
            print(f"  Motor {motor_id}: Read failed")
    
    # Follower arm status
    print_colored("\n🤖 Follower Arm:", Colors.CYAN)
    for motor_id in connected_follower_motors:
        position, angle, success = read_motor_position(
            follower_port_handler, follower_packet_handler, motor_id, FOLLOWER_CONFIG)
        if success:
            offset = position_offsets[motor_id]
            d = direction_multipliers[motor_id]
            print(f"  Motor {motor_id}: {position:4d} ({angle:+6.2f}°) [offset: {offset:+.2f}°, dir: {d:+d}]")
        else:
            print(f"  Motor {motor_id}: Read failed")
    
    # Sync status
    status = "🟢 ACTIVE" if sync_active else "🔴 STOPPED"
    print_colored(f"\n🔄 Sync Status: {status}", Colors.CYAN)

def show_help():
    """Show available commands"""
    print_colored("\n🎮 Leader-Follower Sync Controller - Commands:", Colors.HEADER)
    print_colored("=" * 70, Colors.HEADER)
    print_colored("🔄 Synchronization:", Colors.CYAN)
    print("  'start'              - Start leader→follower synchronization")
    print("  'stop'               - Stop synchronization")
    print("  'status'             - Show current status of both arms")
    
    print_colored("\n⚙️  Calibration:", Colors.CYAN)
    print("  'cal auto'           - Auto-calibrate by moving both arms to zero")
    print("  'cal zero'           - Move both arms to zero position manually") 
    print("  'cal M O'            - Set offset O degrees for motor M")
    print("                         Example: 'cal 1 -5' sets -5° offset for motor 1")
    print("  'cal reset'          - Reset all offsets to 0")
    print("  'cal save'           - Save current calibration (offsets + direction)")
    print("  'cal load'           - Load calibration (offsets + direction) from file")
    print("                        Tip: set direction via zero_calibration_helper.py → 'dir' commands")
    print("  'map L F'            - Map leader motor L to follower motor F (e.g., 'map 2 3')")
    print("  'map reset'          - Reset mapping to identity (1→1, 2→2, 3→3, 4→4)")
    
    print_colored("\n🔧 System:", Colors.CYAN)
    print("  'h' or 'help'       - Show this help")
    print("  'c' or 'clear'      - Clear screen")
    print("  'q' or 'quit'       - Quit controller")
    print_colored("=" * 70, Colors.HEADER)
    print_colored("💡 Usage Tips:", Colors.WARNING)
    print("  1. Start sync with 'start' command")
    print("  2. Move leader arm manually to see follower follow")
    print("  3. Use calibration if motors don't align properly")
    print("  4. Stop sync before quitting with 'stop' command")

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def main():
    global sync_active
    global leader_port_handler, leader_packet_handler
    global follower_port_handler, follower_packet_handler
    global connected_leader_motors, connected_follower_motors
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        global sync_active
        sync_active = False
        print_colored('\n\n🛑 Shutting down...', Colors.WARNING)
        cleanup_and_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Header
    clear_screen()
    print_colored("🤖 Leader-Follower Arm Synchronization Controller", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 70, Colors.HEADER)
    print(f"Leader:   {LEADER_CONFIG['port']} @ {LEADER_CONFIG['baudrate']}")
    print(f"Follower: {FOLLOWER_CONFIG['port']} @ {FOLLOWER_CONFIG['baudrate']}")
    print_colored("=" * 70, Colors.HEADER)
    
    # Initialize leader arm
    leader_port_handler, leader_packet_handler = initialize_dynamixel(LEADER_CONFIG, "Leader")
    if leader_port_handler is None:
        print_colored("❌ Failed to initialize leader arm. Exiting.", Colors.FAIL)
        return
    
    # Initialize follower arm
    follower_port_handler, follower_packet_handler = initialize_dynamixel(FOLLOWER_CONFIG, "Follower")
    if follower_port_handler is None:
        print_colored("❌ Failed to initialize follower arm. Exiting.", Colors.FAIL)
        return
    
    # Ping motors
    connected_leader_motors = ping_motors(leader_port_handler, leader_packet_handler, LEADER_CONFIG, "Leader")
    connected_follower_motors = ping_motors(follower_port_handler, follower_packet_handler, FOLLOWER_CONFIG, "Follower")
    
    if not connected_leader_motors or not connected_follower_motors:
        print_colored("❌ Not enough motors found. Exiting.", Colors.FAIL)
        return
    
    print_colored(f"✅ Leader motors: {connected_leader_motors}", Colors.GREEN)
    print_colored(f"✅ Follower motors: {connected_follower_motors}", Colors.GREEN)
    
    # Show initial status
    show_status()
    
    # Show help
    show_help()
    
    # Main command loop
    while True:
        try:
            if sync_active:
                cmd = input(f"\n{Colors.CYAN}🎯 [SYNCING] Enter command: {Colors.ENDC}").strip().lower()
            else:
                cmd = input(f"\n{Colors.CYAN}🎯 Enter command: {Colors.ENDC}").strip().lower()
            
            if cmd in ['q', 'quit', 'exit']:
                break
                
            elif cmd in ['h', 'help']:
                show_help()
                
            elif cmd in ['c', 'clear']:
                clear_screen()
                print_colored("🤖 Leader-Follower Sync Controller", Colors.HEADER + Colors.BOLD)
                
            elif cmd == 'start':
                start_synchronization()
                
            elif cmd == 'stop':
                stop_synchronization()
                
            elif cmd in ['s', 'status']:
                show_status()
                
            elif cmd.startswith('cal '):
                parts = cmd.split()
                if len(parts) == 2:
                    if parts[1] == 'auto':
                        auto_calibrate()
                    elif parts[1] == 'zero':
                        move_to_zero_position()
                    elif parts[1] == 'reset':
                        for motor_id in MOTOR_IDS:
                            position_offsets[motor_id] = 0.0
                        print_colored("✓ All offsets reset to 0°", Colors.GREEN)
                    elif parts[1] == 'save':
                        save_calibration()
                    elif parts[1] == 'load':
                        load_calibration()
                    else:
                        print_colored("❌ Invalid calibration command. Type 'h' for help.", Colors.FAIL)
                elif len(parts) == 3:
                    try:
                        motor_id = int(parts[1])
                        offset = float(parts[2])
                        calibrate_position(motor_id, offset)
                    except ValueError:
                        print_colored("❌ Invalid format. Use: cal MOTOR_ID OFFSET", Colors.FAIL)
                        print_colored("   Example: cal 1 -5", Colors.FAIL)
                else:
                    print_colored("❌ Invalid calibration format. Type 'h' for help.", Colors.FAIL)
            elif cmd.startswith('map'):
                parts = cmd.split()
                if len(parts) == 2 and parts[1] == 'reset':
                    for l in MOTOR_IDS:
                        id_map[l] = l
                    print_colored("✓ Mapping reset to identity", Colors.GREEN)
                elif len(parts) == 3:
                    try:
                        l = int(parts[1]); f = int(parts[2])
                        if l in MOTOR_IDS and f in MOTOR_IDS:
                            id_map[l] = f
                            print_colored(f"✓ Set mapping L{l} → F{f}", Colors.GREEN)
                        else:
                            print_colored("❌ Usage: map <leader_id 1-4> <follower_id 1-4>", Colors.FAIL)
                    except ValueError:
                        print_colored("❌ Usage: map <leader_id 1-4> <follower_id 1-4>", Colors.FAIL)
                else:
                    print_colored("❌ Invalid 'map' usage", Colors.FAIL)
                    
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
    global sync_active
    
    # Stop synchronization
    if sync_active:
        stop_synchronization()
        time.sleep(0.5)
    
    # Close ports
    if leader_port_handler:
        leader_port_handler.closePort()
        print_colored("🔌 Leader port closed", Colors.GREEN)
    
    if follower_port_handler:
        follower_port_handler.closePort()
        print_colored("🔌 Follower port closed", Colors.GREEN)
    
    print_colored("👋 Leader-Follower Sync Controller stopped", Colors.GREEN)

if __name__ == "__main__":
    main()
