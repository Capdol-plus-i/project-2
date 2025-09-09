#!/usr/bin/env python3
"""
IK-based Follower Arm Mover
Compute inverse kinematics for a 4-DOF arm and command the follower motors.

Conventions
- Joints: [1] base yaw, [2] shoulder pitch, [3] elbow pitch, [4] wrist pitch
- Link lengths (m): L1 (shoulder→elbow), L2 (elbow→wrist), L3 (wrist→EE)
- Coordinates: base at shoulder joint; X forward, Y left, Z up
- Wrist pitch: end-effector pitch in radial-Z plane (deg)

Usage example
  python ik_move_follower.py --x 0.18 --y 0.00 --z 0.06 --wrist-pitch-deg 0

Notes
- Uses dynamixel_sdk directly; reads follower port/baud from hardware_config.json
- If calibration.json exists, uses direction_multipliers to flip joint signs when --use-calibration-dir
- Does NOT apply position_offsets from calibration (those are leader-relative)
"""
import os
import json
import math
import time
import argparse
from typing import Dict

from dynamixel_sdk import *  # type: ignore
from utils.ik import solve_ik_4dof


PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

MOTOR_IDS = [1, 2, 3, 4]
RESOLUTION_DEG_PER_TICK = 0.088
CENTER_TICKS = 2048


def load_hw_config() -> Dict:
    cfg_path = 'hardware_config.json'
    default = {'port': '/dev/ttyACM1', 'baudrate': 1000000}
    if os.path.exists(cfg_path):
        try:
            data = json.load(open(cfg_path, 'r'))
            fol = data.get('robot_arms', {}).get('follower', {})
            return {
                'port': fol.get('port', default['port']),
                'baudrate': fol.get('baudrate', default['baudrate'])
            }
        except Exception:
            pass
    return default


def load_calibration_dirs() -> Dict[int, int]:
    path = 'calibration.json'
    dirs = {1: 1, 2: 1, 3: 1, 4: 1}
    if os.path.exists(path):
        try:
            data = json.load(open(path, 'r'))
            d = data.get('direction_multipliers', {})
            if d:
                for k, v in d.items():
                    dirs[int(k)] = 1 if int(v) >= 0 else -1
        except Exception:
            pass
    return dirs


def angle_to_ticks(angle_deg: float) -> int:
    ticks = int(round(CENTER_TICKS + angle_deg / RESOLUTION_DEG_PER_TICK))
    return max(0, min(4095, ticks))


def main():
    parser = argparse.ArgumentParser(description='IK move follower arm to (x,y,z) with wrist pitch.')
    parser.add_argument('--x', type=float, required=True, help='Target X (meters)')
    parser.add_argument('--y', type=float, required=True, help='Target Y (meters)')
    parser.add_argument('--z', type=float, required=True, help='Target Z (meters)')
    parser.add_argument('--wrist-pitch-deg', type=float, default=0.0, help='End-effector pitch (deg)')
    parser.add_argument('--elbow', choices=['down', 'up'], default='down', help='Elbow configuration')
    parser.add_argument('--L1', type=float, default=0.12, help='Link length L1 (m)')
    parser.add_argument('--L2', type=float, default=0.12, help='Link length L2 (m)')
    parser.add_argument('--L3', type=float, default=0.08, help='Link length L3 (m)')
    parser.add_argument('--use-calibration-dir', action='store_true', help='Use direction_multipliers from calibration.json')
    parser.add_argument('--dry-run', action='store_true', help='Compute only; do not move motors')

    args = parser.parse_args()

    links = {'L1': args.L1, 'L2': args.L2, 'L3': args.L3}
    elbow_up = (args.elbow == 'up')

    base, shoulder, elbow, wrist, reachable = solve_ik_4dof(
        args.x, args.y, args.z, links,
        wrist_pitch_deg=args.wrist_pitch_deg,
        elbow_up=elbow_up,
    )

    print(f"IK solution (deg): base={base:+.2f}, shoulder={shoulder:+.2f}, elbow={elbow:+.2f}, wrist={wrist:+.2f} | reachable={reachable}")

    if not reachable:
        print("⚠️  Target may be out of reach; solution clamped to boundary.")

    dirs = {1: 1, 2: 1, 3: 1, 4: 1}
    if args.use_calibration_dir:
        dirs = load_calibration_dirs()
        print(f"Using direction multipliers from calibration.json: {dirs}")

    # Map joint angles to motor IDs with optional direction
    target_degs = {
        1: dirs[1] * base,
        2: dirs[2] * shoulder,
        3: dirs[3] * elbow,
        4: dirs[4] * wrist,
    }

    print("Target motor angles (deg): " + ", ".join([f"M{m}={a:+.2f}" for m, a in target_degs.items()]))

    if args.dry_run:
        return

    # Init Dynamixel
    hw = load_hw_config()
    port = PortHandler(hw['port'])
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        print(f"❌ Failed to open port {hw['port']}")
        return
    if not port.setBaudRate(hw['baudrate']):
        print(f"❌ Failed to set baudrate {hw['baudrate']}")
        return
    print(f"✓ Connected follower: {hw['port']} @ {hw['baudrate']}")

    # Enable torque for connected motors
    connected = []
    for mid in MOTOR_IDS:
        model, cr, er = packet.ping(port, mid)
        if cr == COMM_SUCCESS:
            connected.append(mid)
    if not connected:
        print("❌ No follower motors found")
        port.closePort()
        return
    print(f"Connected motors: {connected}")

    for mid in connected:
        packet.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 1)

    # Send goal positions
    for mid in connected:
        deg = target_degs.get(mid, 0.0)
        ticks = angle_to_ticks(deg)
        cr, er = packet.write4ByteTxRx(port, mid, ADDR_GOAL_POSITION, ticks)
        if cr == COMM_SUCCESS and er == 0:
            print(f"  ✓ M{mid}: set {deg:+.2f}° → {ticks}")
        else:
            print(f"  ✗ M{mid}: failed")

    print("⏳ Waiting 2s...")
    time.sleep(2.0)

    # Read back positions
    for mid in connected:
        pos, cr, er = packet.read4ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        if cr == COMM_SUCCESS and er == 0:
            ang = (pos - CENTER_TICKS) * RESOLUTION_DEG_PER_TICK
            print(f"  M{mid}: present {pos} ({ang:+.2f}°)")

    port.closePort()

if __name__ == '__main__':
    main()

