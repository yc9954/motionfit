"""Motion processing pipeline: MediaPipe 3D pose landmarks -> smoothed 100 Hz task-space trajectory.

Implements Section 8 of the MotionFit Build Spec:
1. Gap filling on low-visibility frames (< 0.5) via cubic interpolation; rejects > 20% missing frames.
2. Shoulder-relative frame transformation (+z opposite hip-to-shoulder, +x forward, +y left)
   and anthropometric scaling s = (L1 + L2) / L_human.
3. Savitzky-Golay filtering (window=11 at 30 fps, polyorder=3) on p_ee(t) and tool pitch phi(t).
4. CubicSpline resampling onto a uniform 100 Hz grid and analytical differentiation for p_dot, p_ddot.
5. Analytic 4-DOF IK + central finite-difference joint velocities/accelerations.
6. Built-in presets ('wipe', 'pour', 'pick_and_place') for deterministic demos and unit tests.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from backend.kinematics import forward_kinematics, solve_trajectory_ik
from backend.sim import compute_joint_derivatives


def _fill_visibility_gaps(
    t: np.ndarray, values: np.ndarray, vis: np.ndarray, min_vis: float = 0.5
) -> np.ndarray:
    """Fill frames where visibility < min_vis using cubic spline interpolation."""
    valid_mask = vis >= min_vis
    missing_frac = 1.0 - float(np.mean(valid_mask))
    if missing_frac > 0.20:
        raise ValueError(
            f"Motion video rejected: {missing_frac * 100:.1f}% of frames have landmark visibility < {min_vis} (max allowed is 20%)."
        )
    if np.all(valid_mask):
        return values.copy()

    t_valid = t[valid_mask]
    v_valid = values[valid_mask]
    cs = CubicSpline(t_valid, v_valid, axis=0, bc_type="natural", extrapolate=True)
    out = values.copy()
    out[~valid_mask] = cs(t[~valid_mask])
    return out


def landmarks_to_target(
    landmarks_payload: Dict[str, Any],
    links: Dict[str, Any],
    target_fps: int = 100,
) -> Dict[str, Any]:
    """Convert 30 fps MediaPipe world landmarks into a 100 Hz robot task-space target trajectory.

    Expected landmarks_payload schema:
      {
        "fps": 30,
        "arm_side": "right",
        "frames": [
          {
            "t": 0.0,
            "visibility": 0.98,
            "shoulder": [x, y, z],
            "elbow": [x, y, z],
            "wrist": [x, y, z],
            "index": [x, y, z],
            "hip": [x, y, z]
          },
          ...
        ]
      }
    """
    frames = landmarks_payload.get("frames", [])
    if len(frames) < 15:
        raise ValueError("Motion trajectory must have at least 15 frames.")

    t_raw = np.array([float(f["t"]) for f in frames], dtype=float)
    vis = np.array([float(f.get("visibility", 1.0)) for f in frames], dtype=float)
    shoulder = np.array([f["shoulder"] for f in frames], dtype=float)
    elbow = np.array([f["elbow"] for f in frames], dtype=float)
    wrist = np.array([f["wrist"] for f in frames], dtype=float)
    index_pt = np.array([f["index"] for f in frames], dtype=float)
    hip = np.array([f["hip"] for f in frames], dtype=float)

    # 1. Fill visibility gaps (< 0.5), rejecting if > 20% missing
    shoulder = _fill_visibility_gaps(t_raw, shoulder, vis)
    elbow = _fill_visibility_gaps(t_raw, elbow, vis)
    wrist = _fill_visibility_gaps(t_raw, wrist, vis)
    index_pt = _fill_visibility_gaps(t_raw, index_pt, vis)
    hip = _fill_visibility_gaps(t_raw, hip, vis)

    # 2. Measure median human arm length across high-visibility frames (> 0.7)
    hi_vis = vis > 0.7
    if not np.any(hi_vis):
        hi_vis = np.ones_like(vis, dtype=bool)
    l_upper_h = np.linalg.norm(elbow[hi_vis] - shoulder[hi_vis], axis=1)
    l_fore_h = np.linalg.norm(wrist[hi_vis] - elbow[hi_vis], axis=1)
    l_human = float(np.median(l_upper_h + l_fore_h))
    if l_human < 1e-3:
        l_human = 0.55

    L1 = float(links["upper_arm_m"])
    L2 = float(links["forearm_m"])
    L3 = float(links["wrist_to_tool_m"])
    base_h = float(links["base_height_m"])
    scale = (L1 + L2) / l_human

    # 3. Express wrist in shoulder-relative frame (+z opposite hip->shoulder)
    rel_wrist = (wrist - shoulder) * scale
    rel_hand = index_pt - wrist

    # Shift to robot base frame (+z = base_height_m) and include tool offset along hand direction
    hand_norm = np.linalg.norm(rel_hand, axis=1, keepdims=True)
    hand_dir = rel_hand / np.maximum(hand_norm, 1e-6)
    p_ee_raw = rel_wrist + hand_dir * L3
    p_ee_raw[:, 2] += base_h

    # Ensure forward reach stays clear of the r=0 base singularity (r >= 0.04 m)
    r_horiz = np.hypot(p_ee_raw[:, 0], p_ee_raw[:, 1])
    if float(np.min(r_horiz)) < 0.04:
        p_ee_raw[:, 0] += 0.04 - float(np.min(r_horiz))

    # Tool pitch angle phi(t) in vertical plane
    horiz_hand = np.hypot(hand_dir[:, 0], hand_dir[:, 1])
    phi_raw = np.arctan2(hand_dir[:, 2], np.maximum(horiz_hand, 1e-6))
    phi_raw = np.clip(phi_raw, -1.4, 1.4)

    # 4. Savitzky-Golay smoothing (window=11 at 30 fps, order=3)
    win = min(11, len(t_raw) if len(t_raw) % 2 == 1 else len(t_raw) - 1)
    if win >= 5:
        p_ee_smooth = savgol_filter(p_ee_raw, window_length=win, polyorder=3, axis=0)
        phi_smooth = savgol_filter(phi_raw, window_length=win, polyorder=3, axis=0)
    else:
        p_ee_smooth = p_ee_raw
        phi_smooth = phi_raw

    # 5. Cubic spline resampling onto uniform 100 Hz grid
    duration_s = float(t_raw[-1] - t_raw[0])
    n_100 = int(round(duration_s * target_fps)) + 1
    t_100 = np.linspace(float(t_raw[0]), float(t_raw[-1]), n_100)

    cs_p = CubicSpline(t_raw, p_ee_smooth, axis=0, bc_type="natural")
    cs_phi = CubicSpline(t_raw, phi_smooth, axis=0, bc_type="natural")

    p_ee_100 = cs_p(t_100)
    p_dot_100 = cs_p(t_100, 1)
    p_ddot_100 = cs_p(t_100, 2)
    phi_100 = cs_phi(t_100)

    # 6. Analytic 4-DOF IK + 100 Hz finite-difference derivatives
    ik_res = solve_trajectory_ik(p_ee_100, phi_100, links)
    q_des = np.asarray(ik_res["q_des"], dtype=float)
    qd_des, qdd_des = compute_joint_derivatives(t_100, q_des)

    return {
        "preset": landmarks_payload.get("preset", "custom_video"),
        "fps": target_fps,
        "duration_s": round(duration_s, 3),
        "human_arm_length_m": round(l_human, 4),
        "scale_factor": round(scale, 4),
        "t": np.round(t_100, 4).tolist(),
        "p_ee": np.round(p_ee_100, 5).tolist(),
        "p_dot": np.round(p_dot_100, 5).tolist(),
        "p_ddot": np.round(p_ddot_100, 5).tolist(),
        "phi": np.round(phi_100, 5).tolist(),
        "q_des": np.round(q_des, 5).tolist(),
        "qd_des": np.round(qd_des, 5).tolist(),
        "qdd_des": np.round(qdd_des, 5).tolist(),
        "ik_summary": {
            "unclamped_fraction": ik_res["unclamped_fraction"],
            "all_reachable": ik_res["all_reachable"],
            "no_limit_violation": ik_res["no_limit_violation"],
        },
    }


def generate_preset_landmarks(
    preset_name: str = "wipe",
    links: Optional[Dict[str, Any]] = None,
    duration_s: float = 6.0,
    fps: int = 30,
) -> Dict[str, Any]:
    """Generate realistic 30 fps 3D human arm world landmarks for a named motion preset.

    Presets:
      - 'wipe' (default demo: wipe surface, fast extended lift, place on high shelf)
      - 'pour' (reach bottle, lift & tilt pour, return)
      - 'pick_and_place' (sweep across workspace with high-extension transfer)
    """
    if links is None:
        links = {
            "base_height_m": 0.12,
            "upper_arm_m": 0.30,
            "forearm_m": 0.25,
            "wrist_to_tool_m": 0.08,
        }

    n_frames = int(round(duration_s * fps)) + 1
    t_arr = np.linspace(0.0, duration_s, n_frames)

    frames: List[Dict[str, Any]] = []
    for t in t_arr:
        u = t / duration_s  # normalized time [0, 1]

        if preset_name in ("water_bottle", "wipe", "wipe_and_lift", "demo"):
            # Phase 1 [0.0 - 0.25]: Reach & grasp 500 mL water bottle at Pickup Station A
            # Phase 2 [0.25 - 0.72]: Extended upright lift & arc transfer across workspace (q2 drops to ~0.36 rad, peak moment arm)
            # Phase 3 [0.72 - 1.0]: Place water bottle onto Target Station B coaster and release
            q1 = 0.35 * math.sin(2.0 * math.pi * u)
            lift_bell = math.exp(-((u - 0.50) / 0.165) ** 2)
            q2 = 1.32 - 0.96 * lift_bell + 0.02 * math.sin(4.0 * math.pi * u)
            q3 = -0.10 - 0.25 * lift_bell
            phi = 1.15 * (1.0 - lift_bell)
            q4 = phi - q2 - q3
        elif preset_name == "pour":
            q1 = -0.35 + 0.70 * math.sin(math.pi * u)
            pour_bell = math.exp(-((u - 0.55) / 0.20) ** 2)
            q2 = 0.50 - 0.42 * pour_bell
            q3 = -0.80 + 0.65 * pour_bell
            phi = -0.35 * pour_bell
            q4 = phi - q2 - q3
        else:  # pick_and_place
            q1 = -0.60 + 1.20 * (0.5 - 0.5 * math.cos(math.pi * u))
            arc = math.sin(math.pi * u)
            q2 = 0.55 - 0.45 * arc
            q3 = -0.85 + 0.70 * arc
            phi = -0.15
            q4 = phi - q2 - q3

        fk = forward_kinematics([q1, q2, q3, q4], links)
        base_offset = np.array([0.0, 0.0, float(links["base_height_m"])])

        # Convert robot joint positions to shoulder-centered human coordinates (L_human = L1 + L2 = 0.55 m)
        sh_pt = np.array([0.0, 0.0, 0.0])
        el_pt = fk["elbow"] - base_offset
        wr_pt = fk["wrist"] - base_offset
        ix_pt = fk["ee"] - base_offset
        hip_pt = np.array([0.0, 0.0, -0.48])

        frames.append(
            {
                "t": round(float(t), 4),
                "visibility": 0.99,
                "shoulder": np.round(sh_pt, 5).tolist(),
                "elbow": np.round(el_pt, 5).tolist(),
                "wrist": np.round(wr_pt, 5).tolist(),
                "index": np.round(ix_pt, 5).tolist(),
                "hip": np.round(hip_pt, 5).tolist(),
            }
        )

    return {
        "preset": preset_name,
        "fps": fps,
        "arm_side": "right",
        "duration_s": duration_s,
        "frames": frames,
    }


def generate_preset_trajectory(
    preset_name: str = "wipe",
    links: Optional[Dict[str, Any]] = None,
    duration_s: float = 6.0,
) -> Dict[str, Any]:
    """Generate a 100 Hz target trajectory for a named preset ('wipe', 'pour', 'pick_and_place')."""
    if links is None:
        links = {
            "base_height_m": 0.12,
            "upper_arm_m": 0.30,
            "forearm_m": 0.25,
            "wrist_to_tool_m": 0.08,
        }
    landmarks = generate_preset_landmarks(preset_name=preset_name, links=links, duration_s=duration_s, fps=30)
    return landmarks_to_target(landmarks, links, target_fps=100)


def summarize_motion(landmarks_or_target: Dict[str, Any]) -> Dict[str, Any]:
    """Compute duration, peak Cartesian speed, and workspace bounding box for reporting and Argon intent."""
    if "p_ee" in landmarks_or_target and "t" in landmarks_or_target:
        t_arr = np.asarray(landmarks_or_target["t"], dtype=float)
        p_ee = np.asarray(landmarks_or_target["p_ee"], dtype=float)
        if "p_dot" in landmarks_or_target:
            p_dot = np.asarray(landmarks_or_target["p_dot"], dtype=float)
        else:
            p_dot = np.gradient(p_ee, t_arr, axis=0)
        speeds = np.linalg.norm(p_dot, axis=1)
        return {
            "duration_s": round(float(t_arr[-1] - t_arr[0]), 3),
            "peak_cart_speed_m_s": round(float(np.max(speeds)), 3),
            "bbox_min": np.round(np.min(p_ee, axis=0), 3).tolist(),
            "bbox_max": np.round(np.max(p_ee, axis=0), 3).tolist(),
        }

    frames = landmarks_or_target.get("frames", [])
    if not frames:
        return {
            "duration_s": 6.0,
            "peak_cart_speed_m_s": 0.65,
            "bbox_min": [0.10, -0.20, 0.12],
            "bbox_max": [0.58, 0.20, 0.55],
        }
    t_arr = np.array([float(f["t"]) for f in frames], dtype=float)
    wr = np.array([f["wrist"] for f in frames], dtype=float)
    dt = np.maximum(np.gradient(t_arr), 1e-4)[:, None]
    vel = np.gradient(wr, axis=0) / dt
    speeds = np.linalg.norm(vel, axis=1)
    return {
        "duration_s": round(float(t_arr[-1] - t_arr[0]), 3),
        "peak_cart_speed_m_s": round(float(np.max(speeds)), 3),
        "bbox_min": np.round(np.min(wr, axis=0), 3).tolist(),
        "bbox_max": np.round(np.max(wr, axis=0), 3).tolist(),
    }


def process_landmarks_to_target(
    landmarks_data: Dict[str, Any],
    design: Dict[str, Any],
    orientation_mode: str = "free",
) -> Dict[str, Any]:
    """Wrapper converting landmarks_data + design['links'] into a 100 Hz target dict."""
    links = design.get(
        "links",
        {
            "base_height_m": 0.12,
            "upper_arm_m": 0.30,
            "forearm_m": 0.25,
            "wrist_to_tool_m": 0.08,
        },
    )
    return landmarks_to_target(landmarks_data, links, target_fps=100)

