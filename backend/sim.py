"""Two-pass MuJoCo physics verification engine (Pass A: Inverse Dynamics, Pass B: Forward Simulation).

Pass A runs at 100 Hz on the trajectory grid using mj_inverse to obtain required joint torques
and instantaneous torque/speed ratios against the linear DC/BLDC torque-speed curve.
Pass B runs at 500 Hz (dt = 0.002 s) using mj_step with feedforward + PD control clamped to
available_torque(qvel[j]), measuring physical tracking droop (ee_rmse_mm) and actuator saturation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from backend.checker import JOINT_NAMES, evaluate_design
from backend.kinematics import solve_trajectory_ik
from backend.model_builder import (
    available_torque,
    build_mjcf,
    compute_design_mass_and_cost,
    get_joint_actuator_specs,
    load_default_catalog,
    load_default_materials,
)


def compute_joint_derivatives(
    t: np.ndarray, q_des: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 100 Hz joint velocities and accelerations via central finite differences."""
    if len(t) < 2:
        zeros = np.zeros_like(q_des)
        return zeros, zeros
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Trajectory timestamps must be finite and strictly increasing.")
    if not np.all(np.isfinite(q_des)) or len(q_des) != len(t):
        raise ValueError("Joint samples must be finite and match trajectory timestamps.")
    qd_des = np.gradient(q_des, t, axis=0)
    qdd_des = np.gradient(qd_des, t, axis=0)
    return qd_des, qdd_des


def simulate_design(
    design: Dict[str, Any],
    target: Dict[str, Any],
    catalog: Optional[Dict[str, Any]] = None,
    materials: Optional[Dict[str, Any]] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Pass A (100 Hz inverse dynamics) and Pass B (500 Hz torque-clamped forward sim)."""
    if catalog is None:
        catalog = load_default_catalog()
    if materials is None:
        materials = load_default_materials()

    t_arr = np.asarray(target["t"], dtype=float)
    p_ee = np.asarray(target["p_ee"], dtype=float)
    phi = np.asarray(target["phi"], dtype=float)
    n_frames = len(t_arr)
    if (t_arr.ndim != 1 or n_frames < 2 or p_ee.shape != (n_frames, 3)
            or phi.shape != (n_frames,) or not all(np.all(np.isfinite(a)) for a in (t_arr, p_ee, phi))
            or not np.all(np.diff(t_arr) > 0)):
        raise ValueError("Target requires finite t[N], p_ee[N,3], phi[N] with strictly increasing timestamps and N >= 2.")

    # 1. Build MJCF and compile MuJoCo model
    mjcf_xml = build_mjcf(design, catalog, materials)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    # 2. Re-solve analytic IK for current design["links"] against task-space target p_ee(t), phi(t)
    ik_result = solve_trajectory_ik(p_ee, phi, design["links"])
    q_des = np.asarray(ik_result["q_des"], dtype=float)
    qd_des, qdd_des = compute_joint_derivatives(t_arr, q_des)

    # 3. Extract actuator specs per joint
    joint_specs = [
        get_joint_actuator_specs(design["joints"][j], catalog) for j in JOINT_NAMES
    ]
    tau_stall_eff = np.array([s["tau_stall_eff"] for s in joint_specs], dtype=float)
    tau_cont_eff = np.array([s["tau_cont_eff"] for s in joint_specs], dtype=float)
    tau_rated_eff = np.array([s["tau_rated_eff"] for s in joint_specs], dtype=float)
    omega_nl_eff = np.array([s["omega_nl_eff"] for s in joint_specs], dtype=float)

    # -------------------------------------------------------------------------
    # Pass A: Inverse Dynamics at 100 Hz
    # -------------------------------------------------------------------------
    tau_req = np.zeros((n_frames, 4), dtype=float)
    tau_avail_a = np.zeros((n_frames, 4), dtype=float)
    instant_ratio = np.zeros((n_frames, 4), dtype=float)
    ee_ik = np.zeros((n_frames, 3), dtype=float)

    for i in range(n_frames):
        data.qpos[:] = q_des[i]
        data.qvel[:] = qd_des[i]
        data.qacc[:] = qdd_des[i]
        mujoco.mj_inverse(model, data)
        tau_req[i] = data.qfrc_inverse.copy()
        ee_ik[i] = data.site("ee").xpos.copy()

        for j in range(4):
            avail = available_torque(
                float(qd_des[i, j]),
                float(tau_stall_eff[j]),
                float(tau_rated_eff[j]),
                float(omega_nl_eff[j]),
            )
            tau_avail_a[i, j] = avail
            instant_ratio[i, j] = abs(float(tau_req[i, j])) / max(avail, 1e-6)

    # -------------------------------------------------------------------------
    # Pass B: Torque-Limited Forward Simulation at 500 Hz (dt = 0.002 s)
    # -------------------------------------------------------------------------
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_des[0]
    data.qvel[:] = qd_des[0]
    mujoco.mj_forward(model, data)

    sim_dt = float(model.opt.timestep)  # 0.002 s (500 Hz)
    i_nom = np.minimum(0.05, np.maximum(np.asarray(model.dof_M0, dtype=float), 1e-4))
    kp = 40.0 * tau_stall_eff
    kd = np.minimum(2.0 * np.sqrt(kp * i_nom), 0.35 * np.asarray(model.dof_M0, dtype=float) / sim_dt)

    data.time = float(t_arr[0])
    q_act = np.zeros((n_frames, 4), dtype=float)
    qd_act = np.zeros((n_frames, 4), dtype=float)
    ee_act = np.zeros((n_frames, 3), dtype=float)
    tau_cmd_rec = np.zeros((n_frames, 4), dtype=float)
    tau_lim_rec = np.zeros((n_frames, 4), dtype=float)
    saturated_duration = np.zeros(4, dtype=float)

    for i in range(n_frames):
        q_act[i] = data.qpos.copy()
        qd_act[i] = data.qvel.copy()
        ee_act[i] = data.site("ee").xpos.copy()

        # Compute instantaneous control at frame i for recording
        tau_lim_now = np.array(
            [
                available_torque(
                    float(data.qvel[j]),
                    float(tau_stall_eff[j]),
                    float(tau_rated_eff[j]),
                    float(omega_nl_eff[j]),
                )
                for j in range(4)
            ],
            dtype=float,
        )
        tau_raw_now = (
            tau_req[i]
            + kp * (q_des[i] - data.qpos)
            + kd * (qd_des[i] - data.qvel)
        )
        tau_cmd_rec[i] = np.clip(tau_raw_now, -tau_lim_now, tau_lim_now)
        tau_lim_rec[i] = tau_lim_now

        if i == n_frames - 1:
            break

        # Integrate the actual interval, including a shorter final step when
        # samples are not an integer multiple of the 500 Hz controller period.
        interval = float(t_arr[i + 1] - t_arr[i])
        elapsed = 0.0
        while elapsed < interval - 1e-12:
            step_dt = min(sim_dt, interval - elapsed)
            model.opt.timestep = step_dt
            alpha = elapsed / interval
            q_ref = (1.0 - alpha) * q_des[i] + alpha * q_des[i + 1]
            qd_ref = (1.0 - alpha) * qd_des[i] + alpha * qd_des[i + 1]
            tau_ff = (1.0 - alpha) * tau_req[i] + alpha * tau_req[i + 1]

            tau_cmd = tau_ff + kp * (q_ref - data.qpos) + kd * (qd_ref - data.qvel)
            tau_lim = np.array(
                [
                    available_torque(
                        float(data.qvel[j]),
                        float(tau_stall_eff[j]),
                        float(tau_rated_eff[j]),
                        float(omega_nl_eff[j]),
                    )
                    for j in range(4)
                ],
                dtype=float,
            )
            for j in range(4):
                if abs(tau_cmd[j]) >= 0.99 * max(tau_lim[j], 1e-6):
                    saturated_duration[j] += step_dt

            data.ctrl[:] = np.clip(tau_cmd, -tau_lim, tau_lim)
            mujoco.mj_step(model, data)
            elapsed += step_dt
        # mj_step integrates qpos after evaluating positional quantities.
        # Refresh the sites so the recorded Cartesian sample matches q_act.
        mujoco.mj_forward(model, data)
    model.opt.timestep = sim_dt

    # -------------------------------------------------------------------------
    # Aggregate Metrics
    # -------------------------------------------------------------------------
    ee_err_m = np.linalg.norm(ee_act - p_ee, axis=1)
    # Time-weighted integration also handles nonuniform captured sample times.
    integration_weights = np.empty(n_frames)
    integration_weights[0] = (t_arr[1] - t_arr[0]) * 0.5
    integration_weights[-1] = (t_arr[-1] - t_arr[-2]) * 0.5
    if n_frames > 2:
        integration_weights[1:-1] = (t_arr[2:] - t_arr[:-2]) * 0.5
    ee_rmse_mm = float(np.sqrt(np.average(ee_err_m**2, weights=integration_weights)) * 1000.0)
    phi_act = np.sum(q_act[:, 1:4], axis=1)
    phi_error = (phi_act - phi + np.pi) % (2.0 * np.pi) - np.pi
    phi_error_deg = np.rad2deg(np.abs(phi_error))
    tool_pitch_rmse_deg = float(np.sqrt(np.average(phi_error_deg**2, weights=integration_weights)))
    worst_phi_idx = int(np.argmax(phi_error_deg))
    worst_idx = int(np.argmax(ee_err_m))
    ee_max_err_mm = float(ee_err_m[worst_idx] * 1000.0)
    t_worst_ee_s = float(t_arr[worst_idx])

    mass_cost = compute_design_mass_and_cost(design, catalog, materials)

    joints_metrics: Dict[str, Dict[str, Any]] = {}
    for j, j_name in enumerate(JOINT_NAMES):
        idx_peak_ratio = int(np.argmax(instant_ratio[:, j]))
        idx_peak_req = int(np.argmax(np.abs(tau_req[:, j])))
        idx_peak_spd = int(np.argmax(np.abs(qd_des[:, j])))

        peak_ratio = float(instant_ratio[idx_peak_ratio, j])
        peak_req_nm = float( abs(tau_req[idx_peak_req, j]) )
        tau_avail_at_peak = float(tau_avail_a[idx_peak_ratio, j])
        t_peak_s = float(t_arr[idx_peak_ratio])

        rms_req_nm = float(np.sqrt(np.average(tau_req[:, j] ** 2, weights=integration_weights)))
        rms_ratio = float(rms_req_nm / max(tau_cont_eff[j], 1e-6))

        peak_speed = float(np.max(np.abs(qd_des[:, j])))
        speed_ratio = float(peak_speed / max(omega_nl_eff[j], 1e-6))
        t_peak_speed_s = float(t_arr[idx_peak_spd])

        sat_time_s = float(saturated_duration[j])

        joints_metrics[j_name] = {
            "actuator_id": joint_specs[j]["actuator_id"],
            "gearbox_id": joint_specs[j]["gearbox_id"],
            "peak_req_nm": round(peak_req_nm, 4),
            "tau_avail_at_peak_nm": round(tau_avail_at_peak, 4),
            "peak_ratio": round(peak_ratio, 4),
            "t_peak_s": round(t_peak_s, 3),
            "rms_req_nm": round(rms_req_nm, 4),
            "tau_cont_eff_nm": round(float(tau_cont_eff[j]), 4),
            "rms_ratio": round(rms_ratio, 4),
            "peak_speed_rad_s": round(peak_speed, 4),
            "omega_nl_eff_rad_s": round(float(omega_nl_eff[j]), 4),
            "speed_ratio": round(speed_ratio, 4),
            "t_peak_speed_s": round(t_peak_speed_s, 3),
            "saturated_time_s": round(sat_time_s, 3),
            "tau_stall_eff_nm": round(float(tau_stall_eff[j]), 4),
            "continuous_torque_derived": joint_specs[j]["continuous_torque_derived"],
            "catalog_source": joint_specs[j]["catalog_source"],
        }

    links = design["links"]
    reach_m = (
        float(links["upper_arm_m"])
        + float(links["forearm_m"])
        + float(links["wrist_to_tool_m"])
    )

    metrics: Dict[str, Any] = {
        "ee_rmse_mm": round(ee_rmse_mm, 3),
        "orientation_mode": target.get("orientation_mode", "free"),
        "tool_pitch_rmse_deg": round(tool_pitch_rmse_deg, 3),
        "tool_pitch_max_error_deg": round(float(phi_error_deg[worst_phi_idx]), 3),
        "t_worst_tool_pitch_s": round(float(t_arr[worst_phi_idx]), 6),
        "ee_max_err_mm": round(ee_max_err_mm, 3),
        "t_worst_ee_s": round(t_worst_ee_s, 3),
        "total_mass_kg": round(float(mass_cost["total_mass_kg"]), 4),
        "total_cost_usd": round(float(mass_cost["total_cost_usd"]), 2),
        "reach_m": round(reach_m, 4),
        "ik_unclamped_fraction": round(float(ik_result["unclamped_fraction"]), 4),
        "joints": joints_metrics,
        "mass_cost_breakdown": mass_cost,
    }

    check_result = evaluate_design(metrics, design, ik_result=ik_result, intent=intent)

    # Downsample 100 Hz -> 50 Hz (step=2) for trace.json
    step_ds = 2 if n_frames > 20 else 1
    idx_ds = np.unique(np.append(np.arange(0, n_frames, step_ds), n_frames - 1))
    trace: Dict[str, Any] = {
        "fps": round(float((len(idx_ds) - 1) / (t_arr[-1] - t_arr[0])), 3),
        "links": dict(design["links"]),
        "t": np.round(t_arr[idx_ds], 6).tolist(),
        "q_des": np.round(q_des[idx_ds], 4).tolist(),
        "q_act": np.round(q_act[idx_ds], 4).tolist(),
        "qd_des": np.round(qd_des[idx_ds], 4).tolist(),
        "qd_act": np.round(qd_act[idx_ds], 4).tolist(),
        "phi_des": np.round(phi[idx_ds], 5).tolist(),
        "phi_act": np.round(phi_act[idx_ds], 5).tolist(),
        "ee_des": np.round(p_ee[idx_ds], 5).tolist(),
        "ee_ik": np.round(ee_ik[idx_ds], 5).tolist(),
        "ee_error_mm": np.round(ee_err_m[idx_ds] * 1000.0, 3).tolist(),
        "ee_act": np.round(ee_act[idx_ds], 4).tolist(),
        "tau_req": np.round(tau_req[idx_ds], 4).tolist(),
        "tau_cmd": np.round(tau_cmd_rec[idx_ds], 4).tolist(),
        "tau_avail": np.round(tau_lim_rec[idx_ds], 4).tolist(),
    }

    return {
        "mjcf_xml": mjcf_xml,
        "ik_result": {
            "unclamped_fraction": ik_result["unclamped_fraction"],
            "all_reachable": ik_result["all_reachable"],
            "no_limit_violation": ik_result["no_limit_violation"],
        },
        "metrics": metrics,
        "check": check_result,
        "trace": trace,
        "simulation_scope": {
            "model": "rigid_4dof_fixed_payload",
            "verified": ["inverse_dynamics", "torque_speed_limits", "joint_tracking", "target_tracking", "tool_pitch_tracking", "mass_cost_reach"],
            "not_verified": ["grasp_contact", "object_slip", "ground_self_collision", "mounting_compatibility", "thermal_dissipation", "link_stiffness_backlash"],
            "controller_max_timestep_s": sim_dt,
            "simulated_duration_s": round(float(data.time - t_arr[0]), 9),
        },
    }


def main() -> None:
    """CLI entrypoint: `python -m backend.sim demo`."""
    demo_dir = Path(__file__).resolve().parent / "data" / "demo"
    design_path = demo_dir / "design.json"
    target_path = demo_dir / "target.json"
    intent_path = demo_dir / "intent.json"

    if not design_path.exists() or not target_path.exists():
        print(f"Error: Demo files not found in {demo_dir}", file=sys.stderr)
        sys.exit(1)

    design = json.loads(design_path.read_text())
    target = json.loads(target_path.read_text())
    intent = json.loads(intent_path.read_text()) if intent_path.exists() else None

    res = simulate_design(design, target, intent=intent)
    out = {
        "design_id": design.get("design_id"),
        "passed": res["check"]["passed"],
        "metrics": res["metrics"],
        "checks": res["check"]["checks"],
        "violations": res["check"]["violations"],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
