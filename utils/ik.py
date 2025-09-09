#!/usr/bin/env python3
"""
Lightweight geometric inverse kinematics utilities for a 4-DOF arm:
- Joint 1: base yaw (about Z)
- Joint 2: shoulder pitch (planar)
- Joint 3: elbow pitch (planar)
- Joint 4: wrist pitch (planar)

Kinematic assumptions
- Link lengths: L1 (shoulder→elbow), L2 (elbow→wrist), L3 (wrist→EE)
- Base frame: origin at shoulder joint; X forward, Y left, Z up (right-handed)
- Wrist pitch defines the end-effector pitch angle in the radial-Z plane

Return angles in degrees.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def solve_planar_3link(wx: float, wz: float, L1: float, L2: float, L3: float,
                       wrist_pitch_deg: float = 0.0, elbow_up: bool = False) -> Tuple[float, float, float, bool]:
    """
    Solve for shoulder, elbow, wrist given wrist center target (wx, wz) after subtracting L3 along wrist pitch.
    Returns (shoulder_deg, elbow_deg, wrist_deg, reachable)
    """
    # Effective wrist center after considering wrist link orientation
    theta_w = math.radians(wrist_pitch_deg)
    x = wx - L3 * math.cos(theta_w)
    z = wz - L3 * math.sin(theta_w)

    # Law of cosines for elbow
    r2 = x * x + z * z
    c2 = clamp((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
    reachable = (abs(c2) <= 1.0)

    s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow_up:
        s2 = -s2
    theta2 = math.atan2(s2, c2)  # elbow

    # Shoulder
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    theta1 = math.atan2(z, x) - math.atan2(k2, k1)  # shoulder

    # Wrist to achieve requested EE pitch
    wrist = theta_w - (theta1 + theta2)

    return math.degrees(theta1), math.degrees(theta2), math.degrees(wrist), reachable


def solve_ik_4dof(x: float, y: float, z: float, links: Dict[str, float],
                  wrist_pitch_deg: float = 0.0, elbow_up: bool = False) -> Tuple[float, float, float, float, bool]:
    """
    Solve 4-DOF IK for target (x,y,z) and wrist pitch.
    links: {'L1': ..., 'L2': ..., 'L3': ...} in meters
    Returns (base_deg, shoulder_deg, elbow_deg, wrist_deg, reachable)
    """
    L1 = float(links.get('L1', 0.12))
    L2 = float(links.get('L2', 0.12))
    L3 = float(links.get('L3', 0.08))

    base = math.degrees(math.atan2(y, x))
    r = math.hypot(x, y)

    shoulder, elbow, wrist, reachable = solve_planar_3link(r, z, L1, L2, L3, wrist_pitch_deg, elbow_up)

    return base, shoulder, elbow, wrist, reachable

