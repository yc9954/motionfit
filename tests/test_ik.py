"""Unit Test 3: Analytic Inverse Kinematics round-trip accuracy on 1,000 random reachable points."""

from __future__ import annotations

import numpy as np

from backend.kinematics import forward_kinematics, inverse_kinematics, JOINT_LIMITS


def test_ik_round_trip_1000_points():
    """FK(IK(p)) = p within 0.5 mm on 1,000 random reachable points."""
    links = {
        "base_height_m": 0.12,
        "upper_arm_m": 0.30,
        "forearm_m": 0.25,
        "wrist_to_tool_m": 0.08,
        "tube_profile": "al_tube_25x2",
    }
    rng = np.random.default_rng(42)

    tested = 0
    while tested < 1000:
        q1 = rng.uniform(-2.5, 2.5)
        q2 = rng.uniform(-0.3, 2.1)
        q3 = rng.uniform(-2.0, 2.0)
        # Keep away from exact straight-line singularity and ensure phi stays within J4 limits
        if abs(q3) < 0.08:
            continue
        phi = 0.0  # horizontal tool pitch
        q4 = phi - q2 - q3
        if not (JOINT_LIMITS["J4"][0] + 0.05 <= q4 <= JOINT_LIMITS["J4"][1] - 0.05):
            continue

        q_sample = np.array([q1, q2, q3, q4], dtype=float)
        fk = forward_kinematics(q_sample, links)
        target_p = fk["ee"]

        q_sol, reachable, _ = inverse_kinematics(target_p, phi, links, prev_q=q_sample)
        assert reachable, f"Point {target_p} generated from valid q={q_sample} reported unreachable"

        fk_recovered = forward_kinematics(q_sol, links)["ee"]
        err_mm = float(np.linalg.norm(fk_recovered - target_p) * 1000.0)
        assert err_mm < 0.5, f"IK round-trip error {err_mm:.4f} mm exceeds 0.5 mm for {target_p}"
        tested += 1
