"""Unit Test 2: Verify joint axis conventions (+J2 and +J3 lift the arm upward in +z)."""

from __future__ import annotations

import json
from pathlib import Path
import mujoco
import numpy as np

from backend.model_builder import build_mjcf
from backend.kinematics import forward_kinematics


DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"


def test_joint_axis_signs():
    """+J2 raises the end-effector; +J3 raises the forearm relative to the upper arm."""
    with open(DATA_DIR / "demo" / "catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)
    with open(DATA_DIR / "materials.json", "r", encoding="utf-8") as f:
        materials = json.load(f)

    design = {
        "design_id": "test-axes",
        "topology": "4dof_yaw_pitch_pitch_pitch",
        "links": {
            "base_height_m": 0.12,
            "upper_arm_m": 0.30,
            "forearm_m": 0.25,
            "wrist_to_tool_m": 0.08,
            "tube_profile": "al_tube_25x2",
        },
        "joints": {
            "J1": {"actuator_id": "XM430-W350", "gearbox_id": None},
            "J2": {"actuator_id": "XM430-W350", "gearbox_id": None},
            "J3": {"actuator_id": "XM430-W350", "gearbox_id": None},
            "J4": {"actuator_id": "XL430-W250", "gearbox_id": None},
        },
        "payload_kg": 0.4,
    }

    xml_str = build_mjcf(design, catalog, materials)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")

    # Zero pose
    data.qpos[:] = [0.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    ee_zero = data.site_xpos[ee_site_id].copy()
    wrist_zero = data.xpos[wrist_body_id].copy()

    # Check zero pose points along +x at z = base_height
    expected_x = 0.30 + 0.25 + 0.08
    assert np.isclose(ee_zero[0], expected_x, atol=1e-4)
    assert np.isclose(ee_zero[1], 0.0, atol=1e-4)
    assert np.isclose(ee_zero[2], 0.12, atol=1e-4)

    # +J2 raises the end-effector
    data.qpos[:] = [0.0, 0.35, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    ee_pos_j2 = data.site_xpos[ee_site_id].copy()
    assert ee_pos_j2[2] > ee_zero[2] + 0.15, f"Expected +J2 to raise EE, got z={ee_pos_j2[2]} vs {ee_zero[2]}"

    # Compare analytic FK with MuJoCo site_xpos
    fk_pos = forward_kinematics([0.0, 0.35, 0.0, 0.0], design["links"])["ee"]
    assert np.linalg.norm(fk_pos - ee_pos_j2) < 1e-5

    # +J3 raises the forearm (wrist body and EE) relative to upper arm
    data.qpos[:] = [0.0, 0.0, 0.40, 0.0]
    mujoco.mj_forward(model, data)
    wrist_pos_j3 = data.xpos[wrist_body_id].copy()
    ee_pos_j3 = data.site_xpos[ee_site_id].copy()
    assert wrist_pos_j3[2] > wrist_zero[2] + 0.08, f"Expected +J3 to raise wrist, got z={wrist_pos_j3[2]}"
    assert ee_pos_j3[2] > ee_zero[2] + 0.10, f"Expected +J3 to raise EE, got z={ee_pos_j3[2]}"
