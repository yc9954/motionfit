"""Deterministic hardware & simulation checker (Working Rule 3: The deterministic checker is the referee).

Only checker.py decides pass or fail. Evaluates simulation metrics, IK reachability,
and hardware mass/cost/reach constraints against design thresholds, and produces a
ranked list of violations tagged with timestamp and motion intent phase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


JOINT_NAMES = ("J1", "J2", "J3", "J4")


def find_phase_for_timestamp(t_s: float, intent: Optional[Dict[str, Any]]) -> str:
    """Return the motion phase name from intent.json covering timestamp t_s."""
    if not intent or "phases" not in intent:
        return "motion"
    for phase in intent.get("phases", []):
        t_start = float(phase.get("t_start", 0.0))
        t_end = float(phase.get("t_end", 0.0))
        if t_start - 1e-3 <= t_s <= t_end + 1e-3:
            return str(phase.get("name", "motion"))
    # Fallback to closest phase
    phases = intent.get("phases", [])
    if phases:
        closest = min(
            phases,
            key=lambda p: min(abs(t_s - float(p.get("t_start", 0.0))), abs(t_s - float(p.get("t_end", 0.0)))),
        )
        return str(closest.get("name", "motion"))
    return "motion"


def evaluate_design(
    metrics: Dict[str, Any],
    design: Dict[str, Any],
    ik_result: Optional[Dict[str, Any]] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministically evaluate a design's simulation metrics against all constraints and thresholds.

    Returns:
      {
        "passed": bool,
        "checks": {check_name: bool, ...},
        "details": {check_name: {"passed": bool, "value": float, "limit": float, "margin": float, ...}},
        "violations": [ranked list of violation dicts, largest margin first],
      }
    """
    th = design["thresholds"]
    c = design["constraints"]
    links = design["links"]

    reach_m = (
        float(links["upper_arm_m"])
        + float(links["forearm_m"])
        + float(links["wrist_to_tool_m"])
    )
    unclamped_fraction = (
        float(ik_result["unclamped_fraction"])
        if ik_result is not None and "unclamped_fraction" in ik_result
        else float(metrics.get("ik_unclamped_fraction", 1.0))
    )

    checks: Dict[str, bool] = {}
    details: Dict[str, Dict[str, Any]] = {}
    violations: List[Dict[str, Any]] = []

    def record_upper_bound(
        name: str,
        value: float,
        limit: float,
        unit: str,
        joint: Optional[str] = None,
        t_s: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        is_pass = bool(value <= limit + 1e-9)
        margin = (value / max(limit, 1e-9)) - 1.0
        checks[name] = is_pass
        phase = find_phase_for_timestamp(t_s, intent) if t_s is not None else None
        info: Dict[str, Any] = {
            "check": name,
            "passed": is_pass,
            "value": round(float(value), 4),
            "limit": round(float(limit), 4),
            "margin": round(float(margin), 4),
            "unit": unit,
        }
        if joint:
            info["joint"] = joint
        if t_s is not None:
            info["t_s"] = round(float(t_s), 3)
            info["phase"] = phase
        if extra:
            info.update(extra)
        details[name] = info
        if not is_pass:
            violations.append(info)

    # 1. IK Reachability check (lower bound: unclamped_fraction == 1.0)
    ik_pass = bool(unclamped_fraction >= 1.0 - 1e-6)
    ik_margin = 1.0 - unclamped_fraction
    checks["ik_reachable"] = ik_pass
    ik_info: Dict[str, Any] = {
        "check": "ik_reachable",
        "passed": ik_pass,
        "value": round(unclamped_fraction, 4),
        "limit": 1.0,
        "margin": round(ik_margin, 4),
        "unit": "fraction",
    }
    details["ik_reachable"] = ik_info
    if not ik_pass:
        violations.append(ik_info)

    # 2. End-effector tracking RMSE check
    record_upper_bound(
        name="ee_rmse",
        value=float(metrics["ee_rmse_mm"]),
        limit=float(th["ee_rmse_mm"]),
        unit="mm",
        t_s=float(metrics.get("t_worst_ee_s", 0.0)),
        extra={"ee_max_err_mm": round(float(metrics.get("ee_max_err_mm", 0.0)), 2)},
    )

    # 3. Per-joint peak torque, RMS torque, and peak speed checks
    for j in JOINT_NAMES:
        jm = metrics["joints"][j]
        j_lower = j.lower()

        record_upper_bound(
            name=f"{j_lower}_peak_torque",
            value=float(jm["peak_ratio"]),
            limit=float(th["peak_torque_ratio"]),
            unit="ratio",
            joint=j,
            t_s=float(jm.get("t_peak_s", 0.0)),
            extra={
                "peak_req_nm": round(float(jm["peak_req_nm"]), 3),
                "tau_avail_at_peak_nm": round(float(jm["tau_avail_at_peak_nm"]), 3),
                "saturated_time_s": round(float(jm.get("saturated_time_s", 0.0)), 3),
            },
        )
        record_upper_bound(
            name=f"{j_lower}_rms_torque",
            value=float(jm["rms_ratio"]),
            limit=float(th["rms_torque_ratio"]),
            unit="ratio",
            joint=j,
            t_s=float(jm.get("t_peak_s", 0.0)),
            extra={
                "rms_req_nm": round(float(jm["rms_req_nm"]), 3),
                "tau_cont_eff_nm": round(float(jm["tau_cont_eff_nm"]), 3),
            },
        )
        record_upper_bound(
            name=f"{j_lower}_peak_speed",
            value=float(jm["speed_ratio"]),
            limit=float(th["peak_speed_ratio"]),
            unit="ratio",
            joint=j,
            t_s=float(jm.get("t_peak_speed_s", 0.0)),
            extra={
                "peak_speed_rad_s": round(float(jm["peak_speed_rad_s"]), 3),
                "omega_nl_eff_rad_s": round(float(jm["omega_nl_eff_rad_s"]), 3),
            },
        )

    # 4. System-level mass, cost, and reach checks
    record_upper_bound(
        name="total_mass",
        value=float(metrics["total_mass_kg"]),
        limit=float(c["max_total_mass_kg"]),
        unit="kg",
    )
    record_upper_bound(
        name="total_cost",
        value=float(metrics["total_cost_usd"]),
        limit=float(c["max_cost_usd"]),
        unit="USD",
    )

    min_reach_m = float(c["min_reach_m"])
    reach_pass = bool(reach_m + 1e-9 >= min_reach_m)
    reach_margin = (min_reach_m - reach_m) / max(min_reach_m, 1e-9)
    checks["reach"] = reach_pass
    reach_info: Dict[str, Any] = {
        "check": "reach",
        "passed": reach_pass,
        "value": round(reach_m, 4),
        "limit": round(min_reach_m, 4),
        "margin": round(reach_margin, 4),
        "unit": "m",
    }
    details["reach"] = reach_info
    if not reach_pass:
        violations.append(reach_info)

    # Rank violations largest margin first
    violations.sort(key=lambda v: float(v["margin"]), reverse=True)
    passed = all(checks.values())

    return {
        "passed": passed,
        "checks": checks,
        "details": details,
        "violations": violations,
    }
