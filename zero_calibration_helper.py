#!/usr/bin/env python3
"""
Zero Position Calibration Helper
Interactive tool to help manually align leader and follower arms to zero position
"""

import os
import time
import json
import sys
from dynamixel_sdk import *

# Load hardware configuration
def load_hardware_config():
    config_file = 'hardware_config.json'
    default_config = {
        'leader': {'port': '/dev/ttyACM0', 'baudrate': 1000000},
        'follower': {'port': '/dev/ttyACM1', 'baudrate': 1000000}
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
        except:
            pass
    return default_config

hw_config = load_hardware_config()

# Configuration
PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
MOTOR_IDS = [1, 2, 3, 4]
# Per-follower-motor rotation direction mapping: +1 (same) or -1 (inverted)
DIRECTION_MULTIPLIERS = {1: 1, 2: 1, 3: 1, 4: 1}
# Leader→Follower motor ID map (default identity)
ID_MAP = {1: 1, 2: 2, 3: 3, 4: 4}

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def initialize_arm(config, name):
    port_handler = PortHandler(config['port'])
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    if not port_handler.openPort():
        print_colored(f"✗ Failed to open {name} port {config['port']}", Colors.FAIL)
        return None, None, []
    
    if not port_handler.setBaudRate(config['baudrate']):
        print_colored(f"✗ Failed to set {name} baudrate", Colors.FAIL)
        return None, None, []
    
    print_colored(f"✓ {name} arm connected: {config['port']}", Colors.GREEN)
    
    # Find connected motors
    connected = []
    for motor_id in MOTOR_IDS:
        model_number, comm_result, error = packet_handler.ping(port_handler, motor_id)
        if comm_result == COMM_SUCCESS:
            connected.append(motor_id)
    
    print_colored(f"✓ {name} motors found: {connected}", Colors.GREEN)
    return port_handler, packet_handler, connected

def read_position(port_handler, packet_handler, motor_id):
    position, comm_result, error = packet_handler.read4ByteTxRx(
        port_handler, motor_id, ADDR_PRESENT_POSITION)
    
    if comm_result == COMM_SUCCESS and error == 0:
        angle = (position - 2048) * 0.088
        return position, angle, True
    return None, None, False

def disable_torque(port_handler, packet_handler, motor_id):
    comm_result, error = packet_handler.write1ByteTxRx(
        port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
    return comm_result == COMM_SUCCESS and error == 0

def enable_torque(port_handler, packet_handler, motor_id):
    comm_result, error = packet_handler.write1ByteTxRx(
        port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
    return comm_result == COMM_SUCCESS and error == 0

def angle_to_position(angle_deg):
    """Convert angle (deg) to raw position ticks (0-4095) assuming 0.088°/tick and center 2048."""
    ticks = int(2048 + (angle_deg / 0.088))
    return max(0, min(4095, ticks))

def set_goal_position(port_handler, packet_handler, motor_id, angle_deg):
    target = angle_to_position(angle_deg)
    comm_result, error = packet_handler.write4ByteTxRx(
        port_handler, motor_id, ADDR_GOAL_POSITION, target)
    return comm_result == COMM_SUCCESS and error == 0

def show_positions(leader_handler, leader_packet, leader_motors, 
                  follower_handler, follower_packet, follower_motors):
    """Show current positions of both arms"""
    
    print_colored("\n📊 Current Positions:", Colors.HEADER)
    print_colored("-" * 80, Colors.HEADER)
    
    print_colored("🎯 Leader Arm:", Colors.CYAN)
    leader_positions = {}
    for motor_id in leader_motors:
        pos, angle, success = read_position(leader_handler, leader_packet, motor_id)
        if success:
            leader_positions[motor_id] = angle
            print(f"  Motor {motor_id}: {pos:4d} ({angle:+7.2f}°)")
        else:
            print(f"  Motor {motor_id}: Read failed")
    
    print_colored("\n🤖 Follower Arm:", Colors.CYAN)
    follower_positions = {}
    for motor_id in follower_motors:
        pos, angle, success = read_position(follower_handler, follower_packet, motor_id)
        if success:
            follower_positions[motor_id] = angle
            print(f"  Motor {motor_id}: {pos:4d} ({angle:+7.2f}°)")
        else:
            print(f"  Motor {motor_id}: Read failed")
    
    # Show differences with mapping and direction
    print_colored("\n📐 Angle Differences with Mapping (dir*Leader[L] - Follower[F]):", Colors.WARNING)
    for leader_id in MOTOR_IDS:
        follower_id = ID_MAP.get(leader_id)
        if follower_id is None:
            continue
        if leader_id in leader_positions and follower_id in follower_positions:
            d = DIRECTION_MULTIPLIERS.get(follower_id, 1)
            diff = (d * leader_positions[leader_id]) - follower_positions[follower_id]
            status = "✓ GOOD" if abs(diff) < 3.0 else "⚠️  NEEDS ADJUSTMENT"
            print(f"  L{leader_id}→F{follower_id}: {diff:+7.2f}° {status}")
        else:
            print(f"  L{leader_id}→F{follower_id}: Not available on both arms")

def show_help():
    print_colored("\n🎮 Zero Calibration Helper - Commands:", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    print_colored("📊 Status:", Colors.CYAN)
    print("  's' or 'status'     - Show current positions and differences")
    print("  'watch'             - Continuous position monitoring")
    
    print_colored("\n🔧 Setup:", Colors.CYAN)
    print("  'disable'           - Disable all motor torques for manual adjustment")
    print("  'enable'            - Enable follower torques to test moves")
    print("  'guide'             - Show step-by-step zero positioning guide")
    
    print_colored("\n💾 Calibration:", Colors.CYAN)
    print("  'save'              - Save current positions as zero reference (zero_positions.json)")
    print("  'cal'               - Compute offsets (F - dir[F]*L) using mapping and preview")
    print("  'savecal'           - Compute and save offsets+direction+mapping to calibration.json")
    print("  'loadcal'           - Load and show offsets+direction+mapping from calibration.json")
    print("  'apply'             - Apply current offsets (with direction & mapping) to move follower")
    print("  'load'              - Load and show saved zero reference")

    print_colored("\n↔️  Direction:", Colors.CYAN)
    print("  'dir'               - Show current direction mapping per motor")
    print("  'dir M S'           - Set motor M direction to S (1 or -1)")
    print("                         Example: 'dir 2 -1' (invert motor 2)")
    
    print_colored("\n🔀 ID Mapping (Leader→Follower):", Colors.CYAN)
    print("  'map'               - Show current ID mapping (L→F)")
    print("  'map L F'           - Map leader motor L to follower motor F")
    print("  'map reset'         - Reset mapping to identity (1→1, 2→2, 3→3, 4→4)")

    print_colored("\n🔧 System:", Colors.CYAN)
    print("  'h' or 'help'      - Show this help")
    print("  'c' or 'clear'     - Clear screen")
    print("  'q' or 'quit'      - Quit helper")
    print_colored("=" * 60, Colors.HEADER)

def show_guide():
    print_colored("\n📖 Step-by-Step Zero Position Guide:", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    
    print_colored("🔧 Preparation:", Colors.CYAN)
    print("1. Run 'disable' command to turn off all motor torques")
    print("2. You can now move both arms freely by hand")
    print("3. Position both arms in the same pose manually")
    
    print_colored("\n📐 Recommended Zero Positions:", Colors.CYAN)
    print("• Motor 1 (Base):     Point straight forward (0°)")
    print("• Motor 2 (Shoulder): Horizontal or specific angle you prefer")
    print("• Motor 3 (Elbow):    Straight (0°) or 90° bend")
    print("• Motor 4 (Wrist):    Horizontal (0°) or your preferred angle")
    
    print_colored("\n✅ Alignment Process:", Colors.CYAN)
    print("1. Start with Motor 1 (Base) - align both arms to point forward")
    print("2. Adjust Motor 2 (Shoulder) - make both arms at same height")
    print("3. Adjust Motor 3 (Elbow) - align elbow angles")
    print("4. Finally Motor 4 (Wrist) - align end effector orientation")
    
    print_colored("\n🔍 Verification:", Colors.CYAN)
    print("1. Use 'status' command to check angle differences")
    print("2. Manually fine-tune any motor that shows >3° difference")
    print("3. Use 'watch' command for real-time feedback while adjusting")
    print("4. When satisfied, use 'save' to record the zero positions")
    
    print_colored("\n💡 Tips:", Colors.WARNING)
    print("• Take your time - precision here improves all future operations")
    print("• Use 'watch' mode for real-time feedback during adjustment")
    print("• Aim for <3° difference between leader and follower")
    print("• Save multiple calibrations with different names if needed")

def watch_positions(leader_handler, leader_packet, leader_motors, 
                   follower_handler, follower_packet, follower_motors):
    """Real-time position monitoring"""
    print_colored("\n🔄 Real-time Position Monitoring", Colors.CYAN)
    print_colored("📍 Adjust arms manually - Press Enter to stop", Colors.CYAN)
    print_colored("-" * 80, Colors.HEADER)
    
    import select
    import sys
    
    while True:
        # Check if Enter was pressed (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break
        
        # Read and display positions
        print(f"\r{Colors.BLUE}Leader: {Colors.ENDC}", end="")
        for motor_id in leader_motors:
            pos, angle, success = read_position(leader_handler, leader_packet, motor_id)
            if success:
                print(f"M{motor_id}:{angle:+6.1f}° ", end="")
            else:
                print(f"M{motor_id}:---- ", end="")
        
        print(f"| {Colors.GREEN}Follower: {Colors.ENDC}", end="")
        for motor_id in follower_motors:
            pos, angle, success = read_position(follower_handler, follower_packet, motor_id)
            if success:
                print(f"M{motor_id}:{angle:+6.1f}° ", end="")
            else:
                print(f"M{motor_id}:---- ", end="")
        
        print(" " * 10, end="", flush=True)
        time.sleep(0.2)
    
    print(f"\n{Colors.WARNING}🛑 Monitoring stopped{Colors.ENDC}")

def compute_offsets(leader_handler, leader_packet, leader_motors,
                    follower_handler, follower_packet, follower_motors):
    """Compute offsets per follower motor using mapping: offset[F] = follower[F] - dir[F]*leader[L]."""
    offsets = {}
    print_colored("\n🧮 Computing offsets with mapping (F - dir*L)...", Colors.CYAN)
    for leader_id in MOTOR_IDS:
        follower_id = ID_MAP.get(leader_id)
        if follower_id is None:
            continue
        if leader_id in leader_motors and follower_id in follower_motors:
            _, leader_angle, ok_l = read_position(leader_handler, leader_packet, leader_id)
            _, follower_angle, ok_f = read_position(follower_handler, follower_packet, follower_id)
            if ok_l and ok_f:
                d = DIRECTION_MULTIPLIERS.get(follower_id, 1)
                offsets[follower_id] = follower_angle - d * leader_angle
                print(f"  L{leader_id}→F{follower_id}: F {follower_angle:+6.2f}° - (dir {d:+d} * L {leader_angle:+6.2f}°) = {offsets[follower_id]:+6.2f}°")
            else:
                print_colored(f"  L{leader_id}→F{follower_id}: read failed on one side", Colors.WARNING)
        else:
            print_colored(f"  L{leader_id}→F{follower_id}: not available on both arms", Colors.WARNING)
    return offsets

def save_calibration(offsets, filename="calibration.json"):
    """Save offsets to a file compatible with leader_follower_sync.py (position_offsets)."""
    try:
        calibration_data = {
            'timestamp': time.time(),
            'position_offsets': offsets,
            'direction_multipliers': DIRECTION_MULTIPLIERS,
            'id_map': ID_MAP,
            'meta': {
                'note': 'Model: follower_target = dir[F]*leader[L] + offset[F] (degrees) with mapping L→F'
            }
        }
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print_colored(f"💾 Calibration saved to {filename}", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ Failed to save calibration: {e}", Colors.FAIL)
        return False

def apply_offsets(leader_handler, leader_packet, leader_motors,
                  follower_handler, follower_packet, follower_motors,
                  offsets):
    """Apply offsets by commanding follower to leader_angle + offset for each motor."""
    print_colored("\n📤 Applying offsets: moving follower to match leader...", Colors.CYAN)
    # Enable follower torque
    enabled = 0
    for m in follower_motors:
        if enable_torque(follower_handler, follower_packet, m):
            enabled += 1
    if enabled == 0:
        print_colored("❌ Failed to enable follower torque", Colors.FAIL)
        return False
    
    moved = 0
    for leader_id in MOTOR_IDS:
        follower_id = ID_MAP.get(leader_id)
        if follower_id is None:
            continue
        if leader_id in leader_motors and follower_id in follower_motors and follower_id in offsets:
            _, leader_angle, ok = read_position(leader_handler, leader_packet, leader_id)
            if not ok:
                continue
            d = DIRECTION_MULTIPLIERS.get(follower_id, 1)
            target_angle = d * leader_angle + offsets[follower_id]
            if set_goal_position(follower_handler, follower_packet, follower_id, target_angle):
                moved += 1
                print_colored(f"  ✓ L{leader_id}→F{follower_id}: leader {leader_angle:+6.2f}° → follower target {target_angle:+6.2f}°", Colors.GREEN)
    if moved == 0:
        print_colored("⚠️  No moves issued (check connections)", Colors.WARNING)
        return False
    print_colored("⏳ Waiting 2s for motion...", Colors.WARNING)
    time.sleep(2)
    # Quick verification
    print_colored("\n🔍 Verifying follower angles:", Colors.CYAN)
    for follower_id in MOTOR_IDS:
        if follower_id in follower_motors and follower_id in offsets:
            _, angle, ok = read_position(follower_handler, follower_packet, follower_id)
            if ok:
                print(f"  Follower M{follower_id}: {angle:+6.2f}°")
    return True
 
def show_directions():
    print_colored("\n↔️  Current Direction Mapping:", Colors.CYAN)
    for m in MOTOR_IDS:
        s = DIRECTION_MULTIPLIERS[m]
        label = 'same' if s == 1 else 'inverted'
        print(f"  Follower Motor {m}: {s:+d} ({label})")

def load_calibration_file(filename="calibration.json"):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        pos = data.get('position_offsets', {})
        dirs = data.get('direction_multipliers', {})
        mapping = data.get('id_map', {})
        # Normalize keys
        pos = {int(k): v for k, v in pos.items()} if pos else {}
        dirs = {int(k): int(v) for k, v in dirs.items()} if dirs else {}
        print_colored("\n📁 Loaded calibration file:", Colors.GREEN)
        if pos:
            print_colored("Offsets (deg):", Colors.CYAN)
            for m in MOTOR_IDS:
                if m in pos:
                    print(f"  M{m}: {pos[m]:+6.2f}")
        if dirs:
            print_colored("Directions:", Colors.CYAN)
            for m in MOTOR_IDS:
                if m in dirs:
                    print(f"  M{m}: {dirs[m]:+d}")
        if mapping:
            print_colored("Mapping (Leader→Follower):", Colors.CYAN)
            mapping = {int(k): int(v) for k, v in mapping.items()}
            for l in MOTOR_IDS:
                if l in mapping:
                    print(f"  L{l} → F{mapping[l]}")
        # Apply into current session
        if dirs:
            for m in MOTOR_IDS:
                if m in dirs:
                    DIRECTION_MULTIPLIERS[m] = 1 if dirs[m] >= 0 else -1
        if mapping:
            for l in MOTOR_IDS:
                if l in mapping:
                    ID_MAP[l] = mapping[l]
        return True
    except FileNotFoundError:
        print_colored("❌ calibration.json not found", Colors.FAIL)
    except Exception as e:
        print_colored(f"❌ Failed to load calibration: {e}", Colors.FAIL)
    return False

def save_zero_positions(leader_handler, leader_packet, leader_motors, 
                       follower_handler, follower_packet, follower_motors,
                       filename="zero_positions.json"):
    """Save current positions as zero reference"""
    
    zero_data = {
        'timestamp': time.time(),
        'leader': {},
        'follower': {},
        'differences': {}
    }
    
    # Read leader positions
    for motor_id in leader_motors:
        pos, angle, success = read_position(leader_handler, leader_packet, motor_id)
        if success:
            zero_data['leader'][motor_id] = {'position': pos, 'angle': angle}
    
    # Read follower positions
    for motor_id in follower_motors:
        pos, angle, success = read_position(follower_handler, follower_packet, motor_id)
        if success:
            zero_data['follower'][motor_id] = {'position': pos, 'angle': angle}
    
    # Calculate differences
    for motor_id in MOTOR_IDS:
        if motor_id in zero_data['leader'] and motor_id in zero_data['follower']:
            diff = zero_data['leader'][motor_id]['angle'] - zero_data['follower'][motor_id]['angle']
            zero_data['differences'][motor_id] = diff
    
    # Save to file
    try:
        with open(filename, 'w') as f:
            json.dump(zero_data, f, indent=2)
        print_colored(f"💾 Zero positions saved to {filename}", Colors.GREEN)
        
        # Show summary
        print_colored("\n📊 Saved Zero Position Summary:", Colors.CYAN)
        for motor_id, diff in zero_data['differences'].items():
            status = "✓ GOOD" if abs(diff) < 3.0 else "⚠️  CHECK"
            print(f"  Motor {motor_id}: {diff:+7.2f}° difference {status}")
        
        return True
    except Exception as e:
        print_colored(f"❌ Failed to save: {e}", Colors.FAIL)
        return False

def main():
    clear_screen()
    print_colored("🎯 Zero Position Calibration Helper", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)
    
    # Initialize both arms
    print_colored("🔧 Initializing arms...", Colors.CYAN)
    
    leader_handler, leader_packet, leader_motors = initialize_arm(hw_config['leader'], "Leader")
    if not leader_handler:
        print_colored("❌ Failed to initialize leader arm", Colors.FAIL)
        return
    
    follower_handler, follower_packet, follower_motors = initialize_arm(hw_config['follower'], "Follower")
    if not follower_handler:
        print_colored("❌ Failed to initialize follower arm", Colors.FAIL)
        return
    
    print_colored("✅ Both arms initialized successfully!", Colors.GREEN)
    
    # Show initial positions
    show_positions(leader_handler, leader_packet, leader_motors,
                  follower_handler, follower_packet, follower_motors)
    
    # Show help
    show_help()
    
    # Main command loop
    while True:
        try:
            cmd = input(f"\n{Colors.CYAN}🎯 Enter command: {Colors.ENDC}").strip().lower()
            
            if cmd in ['q', 'quit', 'exit']:
                break
            elif cmd in ['h', 'help']:
                show_help()
            elif cmd in ['c', 'clear']:
                clear_screen()
                print_colored("🎯 Zero Position Calibration Helper", Colors.HEADER + Colors.BOLD)
            elif cmd in ['s', 'status']:
                show_positions(leader_handler, leader_packet, leader_motors,
                             follower_handler, follower_packet, follower_motors)
            elif cmd == 'guide':
                show_guide()
            elif cmd == 'disable':
                print_colored("\n⚡ Disabling all motor torques...", Colors.WARNING)
                disabled_count = 0
                for motors, handler, packet in [(leader_motors, leader_handler, leader_packet),
                                               (follower_motors, follower_handler, follower_packet)]:
                    for motor_id in motors:
                        if disable_torque(handler, packet, motor_id):
                            disabled_count += 1
                print_colored(f"✅ {disabled_count} motors disabled - you can now move arms manually", Colors.GREEN)
            elif cmd == 'watch':
                watch_positions(leader_handler, leader_packet, leader_motors,
                               follower_handler, follower_packet, follower_motors)
            elif cmd == 'save':
                save_zero_positions(leader_handler, leader_packet, leader_motors,
                                   follower_handler, follower_packet, follower_motors)
            elif cmd.startswith('dir'):
                parts = cmd.split()
                if len(parts) == 1:
                    show_directions()
                elif len(parts) == 2 and parts[1] in ['save', 'load']:
                    if parts[1] == 'save':
                        # Save current directions with freshly computed offsets
                        offsets = compute_offsets(leader_handler, leader_packet, leader_motors,
                                                  follower_handler, follower_packet, follower_motors)
                        if offsets:
                            save_calibration(offsets)
                    else:
                        load_calibration_file()
                elif len(parts) == 3:
                    try:
                        m = int(parts[1])
                        s = int(parts[2])
                        if m in MOTOR_IDS and s in (-1, 1):
                            DIRECTION_MULTIPLIERS[m] = s
                            print_colored(f"✓ Set motor {m} direction to {s:+d}", Colors.GREEN)
                        else:
                            print_colored("❌ Usage: dir <motor_id 1-4> <1|-1>", Colors.FAIL)
                    except ValueError:
                        print_colored("❌ Usage: dir <motor_id 1-4> <1|-1>", Colors.FAIL)
                else:
                    print_colored("❌ Invalid 'dir' usage", Colors.FAIL)
            elif cmd.startswith('map'):
                parts = cmd.split()
                if len(parts) == 1:
                    print_colored("\nCurrent Mapping (Leader→Follower):", Colors.CYAN)
                    for l in MOTOR_IDS:
                        print(f"  L{l} → F{ID_MAP[l]}")
                elif len(parts) == 2 and parts[1] == 'reset':
                    for l in MOTOR_IDS:
                        ID_MAP[l] = l
                    print_colored("✓ Mapping reset to identity", Colors.GREEN)
                elif len(parts) == 3:
                    try:
                        l = int(parts[1]); f = int(parts[2])
                        if l in MOTOR_IDS and f in MOTOR_IDS:
                            ID_MAP[l] = f
                            print_colored(f"✓ Set mapping L{l} → F{f}", Colors.GREEN)
                        else:
                            print_colored("❌ Usage: map <leader_id 1-4> <follower_id 1-4>", Colors.FAIL)
                    except ValueError:
                        print_colored("❌ Usage: map <leader_id 1-4> <follower_id 1-4>", Colors.FAIL)
                else:
                    print_colored("❌ Invalid 'map' usage", Colors.FAIL)
            elif cmd == 'cal':
                offsets = compute_offsets(leader_handler, leader_packet, leader_motors,
                                          follower_handler, follower_packet, follower_motors)
                if offsets:
                    print_colored("\nPreview offsets (deg):", Colors.CYAN)
                    for m in MOTOR_IDS:
                        if m in offsets:
                            print(f"  M{m}: {offsets[m]:+6.2f}")
            elif cmd == 'savecal':
                offsets = compute_offsets(leader_handler, leader_packet, leader_motors,
                                          follower_handler, follower_packet, follower_motors)
                if offsets:
                    save_calibration(offsets)
            elif cmd == 'loadcal':
                load_calibration_file()
            elif cmd == 'apply':
                offsets = compute_offsets(leader_handler, leader_packet, leader_motors,
                                          follower_handler, follower_packet, follower_motors)
                if offsets:
                    apply_offsets(leader_handler, leader_packet, leader_motors,
                                  follower_handler, follower_packet, follower_motors,
                                  offsets)
            elif cmd == 'load':
                try:
                    with open("zero_positions.json", 'r') as f:
                        data = json.load(f)
                    print_colored("📁 Loaded zero position reference:", Colors.GREEN)
                    for motor_id, diff in data.get('differences', {}).items():
                        motor_id = int(motor_id)
                        status = "✓ GOOD" if abs(diff) < 3.0 else "⚠️  CHECK"
                        print(f"  Motor {motor_id}: {diff:+7.2f}° difference {status}")
                except FileNotFoundError:
                    print_colored("❌ No saved zero positions found", Colors.FAIL)
                except Exception as e:
                    print_colored(f"❌ Failed to load: {e}", Colors.FAIL)
            elif cmd == '':
                continue
            else:
                print_colored("❓ Unknown command. Type 'h' for help.", Colors.WARNING)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print_colored(f"❌ Error: {e}", Colors.FAIL)
    
    # Cleanup
    if leader_handler:
        leader_handler.closePort()
    if follower_handler:
        follower_handler.closePort()
    
    print_colored("👋 Zero calibration helper stopped", Colors.GREEN)

if __name__ == "__main__":
    main()
