"""Unit test 5: Demo failure & fix (Section 7 of MotionFit Build Spec).

Verifies that:
1. demo-v0 (XM430-W350 on J1-J3, XL430-W250 on J4, 0.4 kg payload) fails checker.py
   on j2_peak_torque and shows ee_rmse_mm > 15 in Pass B.
2. Replacing J2 with XM540-W270 (or adding a 2:1 gearbox APEX-PG20-2) passes all checks
   with ee_rmse_mm < 15.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from backend.motion import generate_preset_landmarks, landmarks_to_target
from backend.sim import simulate_design


DEMO_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "demo"


def test_demo_failure_and_fix():
    design_v0 = json.loads((DEMO_DIR / "design.json").read_text())
    intent = json.loads((DEMO_DIR / "intent.json").read_text())

    landmarks = generate_preset_landmarks("wipe", links=design_v0["links"], duration_s=6.0, fps=30)
    target = landmarks_to_target(landmarks, design_v0["links"], target_fps=100)

    # Save generated landmarks.json and target.json for the demo bundle
    (DEMO_DIR / "landmarks.json").write_text(json.dumps(landmarks, indent=2))
    (DEMO_DIR / "target.json").write_text(json.dumps(target, indent=2))

    # 1. Simulate demo-v0: must fail j2_peak_torque and droop with ee_rmse_mm > 15
    res_v0 = simulate_design(design_v0, target, intent=intent)
    assert res_v0["check"]["passed"] is False, "demo-v0 should fail checker.py"
    assert res_v0["check"]["checks"]["j2_peak_torque"] is False, (
        f"demo-v0 should fail j2_peak_torque, got ratio={res_v0['metrics']['joints']['J2']['peak_ratio']}"
    )
    assert res_v0["metrics"]["ee_rmse_mm"] > 15.0, (
        f"demo-v0 should droop with ee_rmse_mm > 15, got {res_v0['metrics']['ee_rmse_mm']}"
    )

    # 2. Upgrade J2 to XM540-W270: must pass all checks with ee_rmse_mm < 15
    design_fix_motor = copy.deepcopy(design_v0)
    design_fix_motor["design_id"] = "demo-v1-xm540"
    design_fix_motor["joints"]["J2"]["actuator_id"] = "XM540-W270"
    res_fix_motor = simulate_design(design_fix_motor, target, intent=intent)
    assert res_fix_motor["check"]["passed"] is True, (
        f"XM540-W270 on J2 should pass all checks, violations: {res_fix_motor['check']['violations']}"
    )
    assert res_fix_motor["metrics"]["ee_rmse_mm"] < 15.0

    # 3. Alternative fix: add 2:1 gearbox (APEX-PG20-2) on J2: must also pass all checks
    design_fix_gb = copy.deepcopy(design_v0)
    design_fix_gb["design_id"] = "demo-v1-gb2"
    design_fix_gb["joints"]["J2"]["gearbox_id"] = "APEX-PG20-2"
    res_fix_gb = simulate_design(design_fix_gb, target, intent=intent)
    assert res_fix_gb["check"]["passed"] is True, (
        f"2:1 gearbox on J2 should pass all checks, violations: {res_fix_gb['check']['violations']}"
    )
    assert res_fix_gb["metrics"]["ee_rmse_mm"] < 15.0
