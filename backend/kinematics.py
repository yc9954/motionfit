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

    # Angles differing by a full turn represent the same orientation. Normalize
    # before checking the bounded mechanical joints; never unwrap past a stop.
    for b in branches_pos + branches_neg:
        b[:] = (b + math.pi) % (2.0 * math.pi) - math.pi
    candidates = [(b, reach_pos) for b in branches_pos] + [(b, reach_neg) for b in branches_neg]
    valid = [(b, reach) for b, reach in candidates if reach and _in_limits(b)]
    reference = np.asarray(prev_q, dtype=float) if prev_q is not None else branches_pos[0]
    if valid:
        chosen, reachable = min(valid, key=lambda item: float(np.linalg.norm(item[0] - reference)))
    else:
        # Prefer the closest achievable FK position when a target cannot be
        # reached, while retaining failure flags for the checker.
        def failure_score(item: tuple[np.ndarray, bool]) -> float:
            q, _ = item
            clipped = np.array([np.clip(a, *JOINT_LIMITS[name]) for a, name in zip(q, JOINT_LIMITS)])
            error = np.linalg.norm(forward_kinematics(clipped, links)["ee"] - np.asarray(target_pos))
            return float(error * 100.0 + 0.01 * np.linalg.norm(q - reference))
        chosen, reachable = min(candidates, key=failure_score)

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
    positions = np.asarray(p_ee, dtype=float)
    pitches = np.asarray(phi, dtype=float)
    n_frames = len(positions)
    if positions.shape != (n_frames, 3) or pitches.shape != (n_frames,) or n_frames == 0:
        raise ValueError("IK requires non-empty p_ee[N,3] and phi[N] arrays.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(pitches)):
        raise ValueError("IK targets must be finite.")
    q_traj = np.zeros((n_frames, 4), dtype=float)
    ik_failures = 0
    clamped_frames = 0
    limit_violations_per_joint = {name: 0 for name in JOINT_LIMITS}
    prev_q: np.ndarray | None = None
    for i in range(n_frames):
        # Use the same forward/over-shoulder branches as single-pose IK.
        q, reachable, has_violation = inverse_kinematics(positions[i], float(pitches[i]), links, prev_q)
        if not reachable:
            ik_failures += 1
        if has_violation:
            for j, name in enumerate(JOINT_LIMITS):
                if np.isclose(q[j], JOINT_LIMITS[name][0]) or np.isclose(q[j], JOINT_LIMITS[name][1]):
                    limit_violations_per_joint[name] += 1
        if not reachable or has_violation:
            clamped_frames += 1
        q_traj[i] = q
        prev_q = q

    return {
        "q": q_traj,
        "q_des": q_traj,
        "ik_failures": ik_failures,
        "limit_violations": limit_violations_per_joint,
        "unclamped_fraction": round(float(n_frames - clamped_frames) / n_frames, 4),
        "all_reachable": ik_failures == 0,
        "no_limit_violation": sum(limit_violations_per_joint.values()) == 0,
    }
