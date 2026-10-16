"""Unit Test 1: Static hold torque verification against analytical hand calculation."""

from __future__ import annotations

import json
from pathlib import Path
import mujoco
import numpy as np

from backend.model_builder import build_mjcf, compute_analytical_j2_static_torque


DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"


def test_static_hold_j2_torque():
    """Arm horizontal (q=0), payload on, zero velocity: simulated J2 torque matches g * sum(m_i * x_i) within 2%."""
    with open(DATA_DIR / "demo" / "catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)
    with open(DATA_DIR / "materials.json", "r", encoding="utf-8") as f:
        materials = json.load(f)

    design = {
        "design_id": "test-static",
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

    data.qpos[:] = [0.0, 0.0, 0.0, 0.0]
    data.qvel[:] = [0.0, 0.0, 0.0, 0.0]
    data.qacc[:] = [0.0, 0.0, 0.0, 0.0]
    mujoco.mj_inverse(model, data)

    sim_j2_torque = abs(float(data.qfrc_inverse[1]))
    hand_j2_torque = compute_analytical_j2_static_torque(design, catalog, materials)

    rel_err = abs(sim_j2_torque - hand_j2_torque) / hand_j2_torque
    assert rel_err < 0.02, (
        f"Simulated J2 static torque {sim_j2_torque:.4f} N·m differs from "
        f"analytical {hand_j2_torque:.4f} N·m by {rel_err * 100:.2f}% (> 2%)"
    )
