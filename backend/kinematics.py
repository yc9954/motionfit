"""Analytic Forward and Inverse Kinematics for the 4-DOF serial arm."""

from __future__ import annotations

import math
from typing import Any
import numpy as np


JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "J1": (-2.97, 2.97),
    "J2": (-0.52, 2.62),
    "J3": (-2.62, 2.62),
    "J4": (-2.09, 2.09),
}


def forward_kinematics(q: list[float] | np.ndarray, links: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute 3D positions of base, shoulder, elbow, wrist, and end-effector in world frame.

    Convention (matches arm4.xml.j2):
      - J1: hinge about +z at base
      - J2, J3, J4: hinges about -y (so positive angle lifts upward in +z)
      - Zero pose: arm extended along +x at z = base_height_m
    """
    q1, q2, q3, q4 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    base_h = float(links["base_height_m"])
    L1 = float(links["upper_arm_m"])
    L2 = float(links["forearm_m"])
    L3 = float(links["wrist_to_tool_m"])

    c1, s1 = math.cos(q1), math.sin(q1)
    q23 = q2 + q3
    q234 = q2 + q3 + q4

    r_shoulder = 0.0
    z_shoulder = base_h

    r_elbow = L1 * math.cos(q2)
    z_elbow = z_shoulder + L1 * math.sin(q2)

    r_wrist = r_elbow + L2 * math.cos(q23)
    z_wrist = z_elbow + L2 * math.sin(q23)

    r_ee = r_wrist + L3 * math.cos(q234)
    z_ee = z_wrist + L3 * math.sin(q234)

    return {
        "base": np.array([0.0, 0.0, 0.0], dtype=float),
        "shoulder": np.array([r_shoulder * c1, r_shoulder * s1, z_shoulder], dtype=float),
        "elbow": np.array([r_elbow * c1, r_elbow * s1, z_elbow], dtype=float),
        "wrist": np.array([r_wrist * c1, r_wrist * s1, z_wrist], dtype=float),
        "ee": np.array([r_ee * c1, r_ee * s1, z_ee], dtype=float),
    }


def _in_limits(q: np.ndarray, tol: float = 1e-3) -> bool:
    for idx, name in enumerate(("J1", "J2", "J3", "J4")):
        lo, hi = JOINT_LIMITS[name]
        if q[idx] < lo - tol or q[idx] > hi + tol:
            return False
    return True


def _solve_2link_branches(
    r_w: float, z_w: float, L1: float, L2: float, q1: float, phi: float
) -> tuple[list[np.ndarray], bool]:
    """Return candidate [q1, q2, q3, q4] branches for wrist center (r_w, z_w)."""
    d2 = r_w * r_w + z_w * z_w
    d = math.sqrt(d2)
    max_2link = L1 + L2
    min_2link = abs(L1 - L2)

    reachable = True
    if d > max_2link - 1e-7:
        reachable = d <= max_2link + 1e-4
        scale = (max_2link - 1e-6) / max(d, 1e-9)
        r_w *= scale
        z_w *= scale
        d2 = r_w * r_w + z_w * z_w
    elif d < min_2link + 1e-7:
        reachable = d >= min_2link - 1e-4
        if d > 1e-9:
            scale = (min_2link + 1e-6) / d
            r_w *= scale
            z_w *= scale
        else:
            r_w = min_2link + 1e-6
            z_w = 0.0
        d2 = r_w * r_w + z_w * z_w

    cos_q3 = (d2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    sin_q3_abs = math.sqrt(max(0.0, 1.0 - cos_q3 * cos_q3))

    branches: list[np.ndarray] = []
    # Branch 0: elbow-up (q3 <= 0 means upper arm pitched higher, elbow above line)
    # Branch 1: elbow-down (q3 >= 0)
    for q3 in (-math.atan2(sin_q3_abs, cos_q3), math.atan2(sin_q3_abs, cos_q3)):
        k1 = L1 + L2 * math.cos(q3)
        k2 = L2 * math.sin(q3)
        q2 = math.atan2(z_w, r_w) - math.atan2(k2, k1)
        q4 = phi - q2 - q3
        branches.append(np.array([q1, q2, q3, q4], dtype=float))

    return branches, reachable


def inverse_kinematics(
    target_pos: list[float] | np.ndarray,
    phi: float,
    links: dict[str, Any],
    prev_q: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, bool, bool]:
    """Analytic 4-DOF IK.

    Returns:
      (q_sol, is_reachable, has_limit_violation)
    """
    px, py, pz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    base_h = float(links["base_height_m"])
    L1 = float(links["upper_arm_m"])
    L2 = float(links["forearm_m"])
    L3 = float(links["wrist_to_tool_m"])

    r_mag = math.hypot(px, py)
    z = pz - base_h
    z_w = z - L3 * math.sin(phi)

    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    if r_mag < 1e-6 and prev_q is not None:
        q1_primary = float(prev_q[0])
    else:
        q1_primary = math.atan2(py, px)
    q1_flipped = _wrap_pi(q1_primary + math.pi)

    # Primary forward reach (+r_mag)
    branches_pos, reach_pos = _solve_2link_branches(
        r_mag - L3 * math.cos(phi), z_w, L1, L2, q1_primary, phi
    )
    # Over-the-shoulder reach (-r_mag with flipped J1)
    branches_neg, reach_neg = _solve_2link_branches(
        -r_mag - L3 * math.cos(phi), z_w, L1, L2, q1_flipped, phi
    )

    valid_pos = [b for b in branches_pos if reach_pos and _in_limits(b)]
    valid_neg = [b for b in branches_neg if reach_neg and _in_limits(b)]

    if prev_q is not None:
        prev_arr = np.asarray(prev_q, dtype=float)
        all_valid = valid_pos + valid_neg
        if all_valid:
            chosen = min(all_valid, key=lambda b: float(np.linalg.norm(b - prev_arr)))
            reachable = True
        else:
            pool = branches_pos if reach_pos else (branches_pos + branches_neg)
            chosen = min(pool, key=lambda b: float(np.linalg.norm(b - prev_arr)))
            reachable = reach_pos or reach_neg
    else:
        # Frame 0: prefer forward elbow-up (branches_pos[0]) if valid
        if valid_pos:
            chosen = valid_pos[0]
            reachable = True
        elif valid_neg:
            chosen = valid_neg[0]
            reachable = True
        else:
            chosen = branches_pos[0]
            reachable = reach_pos

    limit_violation = not _in_limits(chosen)
    if limit_violation:
        # Clamp to mechanical joint limits for simulation while flagging violation
        clamped = chosen.copy()
        for idx, name in enumerate(("J1", "J2", "J3", "J4")):
            lo, hi = JOINT_LIMITS[name]
            clamped[idx] = float(np.clip(clamped[idx], lo, hi))
        chosen = clamped

    return chosen, reachable, limit_violation


def solve_trajectory_ik(
    p_ee: np.ndarray, phi: np.ndarray, links: dict[str, Any]
) -> dict[str, Any]:
    """Solve analytic IK across an entire trajectory at N frames."""
    n_frames = len(p_ee)
    q_traj = np.zeros((n_frames, 4), dtype=float)
    ik_failures = 0
    clamped_frames = 0
    limit_violations_per_joint = {"J1": 0, "J2": 0, "J3": 0, "J4": 0}

    prev_q: np.ndarray | None = None
    for i in range(n_frames):
        px, py, pz = float(p_ee[i, 0]), float(p_ee[i, 1]), float(p_ee[i, 2])
        r = math.hypot(px, py)
        z = pz - float(links["base_height_m"])
        r_w = r - float(links["wrist_to_tool_m"]) * math.cos(float(phi[i]))
        z_w = z - float(links["wrist_to_tool_m"]) * math.sin(float(phi[i]))
        q1 = math.atan2(py, px) if r >= 1e-6 else (float(prev_q[0]) if prev_q is not None else 0.0)

        branches, reachable = _solve_2link_branches(
            r_w, z_w, float(links["upper_arm_m"]), float(links["forearm_m"]), q1, float(phi[i])
        )
        if not reachable:
            ik_failures += 1

        valid_branches = [b for b in branches if _in_limits(b)]
        if prev_q is not None:
            pool = valid_branches if valid_branches else branches
            raw_choice = min(pool, key=lambda b: float(np.linalg.norm(b[1:4] - prev_q[1:4])))
        else:
            raw_choice = valid_branches[0] if valid_branches else branches[0]

        has_viol = False
        for idx, name in enumerate(("J1", "J2", "J3", "J4")):
            lo, hi = JOINT_LIMITS[name]
            if raw_choice[idx] < lo - 1e-3 or raw_choice[idx] > hi + 1e-3:
                limit_violations_per_joint[name] += 1
                has_viol = True
            raw_choice[idx] = float(np.clip(raw_choice[idx], lo, hi))
        if not reachable or has_viol:
            clamped_frames += 1

        q_traj[i] = raw_choice
        prev_q = raw_choice

    unclamped_fraction = float(n_frames - clamped_frames) / max(n_frames, 1)
    return {
        "q": q_traj,
        "q_des": q_traj,
        "ik_failures": ik_failures,
        "limit_violations": limit_violations_per_joint,
        "unclamped_fraction": round(unclamped_fraction, 4),
        "all_reachable": ik_failures == 0,
        "no_limit_violation": sum(limit_violations_per_joint.values()) == 0,
    }
