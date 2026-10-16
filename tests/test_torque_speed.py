"""Unit Test 4: Linear torque-speed curve and gearbox transformation checks."""

from __future__ import annotations

import numpy as np

from backend.model_builder import available_torque, get_joint_actuator_specs


def test_torque_speed_curve_at_no_load():
    """At omega = omega_0, available torque is 0; at omega = 0, available torque is tau_s."""
    tau_s = 10.6
    omega_0 = 3.14159

    assert np.isclose(available_torque(0.0, tau_s, omega_0), tau_s)
    assert np.isclose(available_torque(omega_0, tau_s, omega_0), 0.0)
    assert np.isclose(available_torque(-omega_0, tau_s, omega_0), 0.0)
    assert np.isclose(available_torque(1.5 * omega_0, tau_s, omega_0), 0.0)
    assert np.isclose(available_torque(0.5 * omega_0, tau_s, omega_0), 0.5 * tau_s)


def test_gearbox_torque_speed_scaling():
    """Gearbox multiplies stall/continuous torque by ratio*efficiency and divides speed by ratio."""
    catalog = {
        "parts": [
            {
                "part_id": "XM430-W350",
                "type": "actuator",
                "stall_torque_nm": {"value": 4.1},
                "continuous_torque_nm": {"value": 1.35},
                "no_load_speed_rad_s": {"value": 4.817},
                "mass_kg": {"value": 0.082},
                "price_usd": {"value": 310.0},
            },
            {
                "part_id": "APEX-PG20-2",
                "type": "gearbox",
                "ratio": {"value": 2.0},
                "efficiency": {"value": 0.92},
                "rated_output_torque_nm": {"value": 18.0},
                "mass_kg": {"value": 0.065},
                "price_usd": {"value": 115.0},
            },
        ]
    }
    spec = get_joint_actuator_specs({"actuator_id": "XM430-W350", "gearbox_id": "APEX-PG20-2"}, catalog)
    assert np.isclose(spec["tau_s"], 4.1 * 2.0 * 0.92)
    assert np.isclose(spec["tau_cont"], 1.35 * 2.0 * 0.92)
    assert np.isclose(spec["omega_0"], 4.817 / 2.0)
    assert np.isclose(spec["mass_kg"], 0.082 + 0.065)
    assert np.isclose(spec["price_usd"], 310.0 + 115.0)
