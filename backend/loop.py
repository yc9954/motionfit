"""Design iteration loop orchestrating MuJoCo simulation, checker.py, validate.py, and Gemini 4 Argon (Section 5.4)."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional

from backend.gemini import propose_design_changes, repair_design_proposal
from backend.model_builder import load_default_catalog, load_default_materials
from backend.sim import simulate_design
from backend.validate import apply_changes_to_design, validate_proposal


EventCallback = Callable[[Dict[str, Any]], None]


def _compute_prediction_vs_measured(
    predicted_effect: Optional[Dict[str, Any]],
    measured_metrics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Compare Argon's predicted_effect from iteration k-1 against measured MuJoCo metrics in iteration k."""
    if not predicted_effect:
        return None
    j2_m = measured_metrics.get("joints", {}).get("J2", {})
    measured = {
        "j2_peak_ratio": float(j2_m.get("peak_ratio", 0.0)),
        "j2_rms_ratio": float(j2_m.get("rms_ratio", 0.0)),
        "total_mass_kg": float(measured_metrics.get("total_mass_kg", 0.0)),
        "total_cost_usd": float(measured_metrics.get("total_cost_usd", 0.0)),
    }
    comparison: Dict[str, Any] = {}
    for key, meas_val in measured.items():
        pred_val = float(predicted_effect.get(key, 0.0))
        abs_err = round(abs(pred_val - meas_val), 4)
        rel_err_pct = round(abs(pred_val - meas_val) / max(abs(meas_val), 1e-6) * 100.0, 2)
        comparison[key] = {
            "predicted": round(pred_val, 4),
            "measured": round(meas_val, 4),
            "abs_error": abs_err,
            "rel_error_pct": rel_err_pct,
        }
    return comparison


def evaluate_naive_baseline(
    initial_design: Dict[str, Any],
    initial_check: Dict[str, Any],
    target: Dict[str, Any],
    catalog: Dict[str, Any],
    materials: Dict[str, Any],
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stretch Goal 1: Naive baseline ('upgrade every failing joint to the strongest motor in catalog')."""
    naive_design = copy.deepcopy(initial_design)
    naive_design["design_id"] = f"{initial_design.get('design_id', 'demo')}-naive-strongest"

    # Find strongest actuator in catalog by stall torque
    actuators = [p for p in catalog.get("parts", []) if p.get("type") == "actuator"]
    strongest = max(actuators, key=lambda p: float((p.get("stall_torque_nm") or {}).get("value", 0.0)))
    strongest_id = strongest["part_id"]

    failing_joints = {
        v.get("joint")
        for v in initial_check.get("violations", [])
        if v.get("joint") in ("J1", "J2", "J3", "J4")
    }
    if not failing_joints:
        failing_joints = {"J2"}

    for j_name in sorted(failing_joints):
        naive_design["joints"][j_name]["actuator_id"] = strongest_id
        naive_design["joints"][j_name]["gearbox_id"] = None

    sim_res = simulate_design(
        naive_design,
        target,
        catalog=catalog,
        materials=materials,
        intent=intent,
    )
    return {
        "design_id": naive_design["design_id"],
        "upgraded_joints": sorted(failing_joints),
        "actuator_id": strongest_id,
        "passed": sim_res["check"]["passed"],
        "metrics": sim_res["metrics"],
        "check": sim_res["check"],
        "summary": (
            f"Naive baseline swaps {', '.join(sorted(failing_joints))} to strongest catalog motor {strongest_id} "
            f"({strongest['stall_torque_nm']['value']} N·m stall, {strongest['mass_kg']['value']} kg, "
            f"${strongest['price_usd']['value']:.0f}): results in total_mass={sim_res['metrics']['total_mass_kg']:.3f} kg, "
            f"total_cost=${sim_res['metrics']['total_cost_usd']:.2f}, ee_rmse={sim_res['metrics']['ee_rmse_mm']:.1f} mm, "
            f"passed={sim_res['check']['passed']}."
        ),
    }


def run_design_loop(
    initial_design: Dict[str, Any],
    target: Dict[str, Any],
    catalog: Optional[Dict[str, Any]] = None,
    materials: Optional[Dict[str, Any]] = None,
    intent: Optional[Dict[str, Any]] = None,
    max_iterations: int = 5,
    optimize_after_pass: bool = False,
    include_naive_baseline: bool = True,
    on_event: Optional[EventCallback] = None,
) -> Dict[str, Any]:
    """Execute the MotionFit verification & redesign loop (up to max_iterations=5).

    Emits progress events via `on_event` for SSE streaming.
    Returns full loop bundle including iterations, initial/final designs, traces, and MJCF XML.
    """
    if catalog is None:
        catalog = load_default_catalog()
    if materials is None:
        materials = load_default_materials()

    def emit(event_type: str, payload: Dict[str, Any]) -> None:
        if on_event is not None:
            on_event({"event": event_type, "timestamp": round(time.time(), 3), **payload})

    base_design_id = str(initial_design.get("design_id", "demo-v0")).split("-iter")[0]
    current_design = copy.deepcopy(initial_design)
    iterations: List[Dict[str, Any]] = []
    interaction_id: Optional[str] = None
    last_predicted_effect: Optional[Dict[str, Any]] = None
    did_post_pass_opt = False

    emit(
        "loop_started",
        {
            "design_id": current_design.get("design_id"),
            "max_iterations": max_iterations,
            "optimize_after_pass": optimize_after_pass,
        },
    )

    for k in range(max_iterations):
        t_iter_start = time.time()
        emit(
            "simulating",
            {
                "iteration": k,
                "design_id": current_design.get("design_id"),
                "joints": copy.deepcopy(current_design.get("joints")),
                "links": copy.deepcopy(current_design.get("links")),
            },
        )

        sim_res = simulate_design(
            current_design,
            target,
            catalog=catalog,
            materials=materials,
            intent=intent,
        )
        check_res = sim_res["check"]
        metrics = sim_res["metrics"]
        pred_vs_meas = _compute_prediction_vs_measured(last_predicted_effect, metrics)

        iter_record: Dict[str, Any] = {
            "iteration": k,
            "design_id": current_design.get("design_id"),
            "design": copy.deepcopy(current_design),
            "mjcf_xml": sim_res["mjcf_xml"],
            "metrics": metrics,
            "check": check_res,
            "trace": sim_res["trace"],
            "prediction_vs_measured": pred_vs_meas,
            "proposal": None,
            "thought_summary": None,
            "validation": None,
            "proposal_source": None,
            "elapsed_s": round(time.time() - t_iter_start, 3),
        }
        iterations.append(iter_record)

        emit(
            "iteration_simulated",
            {
                "iteration": k,
                "design_id": current_design.get("design_id"),
                "passed": check_res["passed"],
                "ee_rmse_mm": metrics["ee_rmse_mm"],
                "total_mass_kg": metrics["total_mass_kg"],
                "total_cost_usd": metrics["total_cost_usd"],
                "violations": check_res["violations"],
                "checks": check_res["checks"],
                "joints": metrics["joints"],
                "prediction_vs_measured": pred_vs_meas,
            },
        )

        # Check termination condition
        if check_res["passed"]:
            if optimize_after_pass and not did_post_pass_opt and k + 1 < max_iterations:
                did_post_pass_opt = True
                emit(
                    "argon_thinking",
                    {
                        "iteration": k,
                        "mode": "post_pass_optimization",
                        "message": "Current candidate passed. Testing an optional cost or mass improvement; only measured passing candidates can be selected.",
                    },
                )
            else:
                emit(
                    "loop_converged",
                    {
                        "iteration": k,
                        "design_id": current_design.get("design_id"),
                        "passed": True,
                        "ee_rmse_mm": metrics["ee_rmse_mm"],
                        "total_mass_kg": metrics["total_mass_kg"],
                        "total_cost_usd": metrics["total_cost_usd"],
                    },
                )
                break

        if k + 1 >= max_iterations:
            emit(
                "loop_terminated",
                {
                    "iteration": k,
                    "passed": check_res["passed"],
                    "reason": f"Reached maximum iterations ({max_iterations}).",
                },
            )
            break

        # Ask Gemini 4 Argon to diagnose and propose changes
        if not did_post_pass_opt:
            emit(
                "argon_thinking",
                {
                    "iteration": k,
                    "mode": "failure_diagnosis",
                    "violations_count": len(check_res["violations"]),
                    "message": f"Evaluating {len(check_res['violations'])} measured violations against the reviewed catalog.",
                },
            )

        argon_res = propose_design_changes(
            design=current_design,
            check_result=check_res,
            metrics=metrics,
            catalog=catalog,
            intent=intent,
            previous_interaction_id=interaction_id,
            is_first_turn=(interaction_id is None),
            optimize_mode=did_post_pass_opt,
        )
        interaction_id = argon_res.get("interaction_id") if argon_res.get("used_live_api") else None
        proposal = argon_res["proposal"]
        thought_summary = argon_res["thought_summary"]
        iter_record["proposal_source"] = {
            "used_live_api": bool(argon_res.get("used_live_api")),
            "model": argon_res.get("model"),
            "source": argon_res.get("source", "model" if argon_res.get("used_live_api") else "deterministic_catalog_search"),
        }

        emit(
            "argon_proposed",
            {
                "iteration": k,
                "interaction_id": interaction_id,
                "thought_summary": thought_summary,
                "diagnosis": proposal.get("diagnosis"),
                "changes": proposal.get("changes"),
                "predicted_effect": proposal.get("predicted_effect"),
                **iter_record["proposal_source"],
            },
        )

        if not proposal.get("changes"):
            iter_record["proposal"] = proposal
            iter_record["thought_summary"] = thought_summary
            iter_record["elapsed_s"] = round(time.time() - t_iter_start, 3)
            emit("no_feasible_proposal", {"iteration": k, "reason": proposal.get("diagnosis", "No feasible additional proposal was found in this bounded search.")})
            break

        # Validate proposal with deterministic referee validate.py
        val_res = validate_proposal(proposal, current_design, catalog, materials)
        if not val_res["valid"]:
            emit(
                "proposal_validation_failed",
                {
                    "iteration": k,
                    "errors": val_res["errors"],
                    "message": "Proposal failed validate.py; requesting repair on same interaction_id...",
                },
            )
            repaired_res = repair_design_proposal(
                validation_errors=val_res["errors"],
                previous_interaction_id=interaction_id,
                design=current_design,
                check_result=check_res,
                metrics=metrics,
                catalog=catalog,
                intent=intent,
            )
            interaction_id = repaired_res.get("interaction_id") if repaired_res.get("used_live_api") else None
            proposal = repaired_res["proposal"]
            thought_summary = repaired_res["thought_summary"]
            iter_record["proposal_source"] = {"used_live_api": bool(repaired_res.get("used_live_api")), "model": repaired_res.get("model"), "source": repaired_res.get("source", "model" if repaired_res.get("used_live_api") else "deterministic_catalog_search")}
            val_res = validate_proposal(proposal, current_design, catalog, materials)

            if not val_res["valid"]:
                emit(
                    "loop_aborted",
                    {
                        "iteration": k,
                        "errors": val_res["errors"],
                        "reason": "Repair proposal still failed deterministic validation.",
                    },
                )
                iter_record["proposal"] = proposal
                iter_record["thought_summary"] = thought_summary
                iter_record["validation"] = val_res
                break

        iter_record["proposal"] = proposal
        iter_record["thought_summary"] = thought_summary
        iter_record["validation"] = {
            "valid": val_res["valid"],
            "errors": val_res["errors"],
            "computed_summary": val_res["computed_summary"],
        }
        iter_record["elapsed_s"] = round(time.time() - t_iter_start, 3)

        emit(
            "proposal_validated",
            {
                "iteration": k,
                "valid": True,
                "changes": proposal.get("changes"),
                "computed_mass_kg": val_res["computed_summary"]["total_mass_kg"],
                "computed_cost_usd": val_res["computed_summary"]["total_cost_usd"],
            },
        )

        last_predicted_effect = proposal.get("predicted_effect")
        next_design_id = f"{base_design_id}-v{k + 1}"
        current_design = apply_changes_to_design(
            current_design,
            proposal.get("changes", []),
            next_design_id=next_design_id,
        )

    # Select best passing iteration (or final iteration if none passed)
    passing_iters = [it for it in iterations if it["check"]["passed"]]
    if passing_iters:
        best_iter = min(
            passing_iters,
            key=lambda it: (it["metrics"]["total_cost_usd"], it["metrics"]["total_mass_kg"]),
        )
    else:
        best_iter = iterations[-1]

    naive_baseline = None
    if include_naive_baseline and iterations:
        naive_baseline = evaluate_naive_baseline(
            initial_design=iterations[0]["design"],
            initial_check=iterations[0]["check"],
            target=target,
            catalog=catalog,
            materials=materials,
            intent=intent,
        )

    result_bundle = {
        "passed": best_iter["check"]["passed"],
        "iterations_count": len(iterations),
        "initial_design": iterations[0]["design"],
        "final_design": best_iter["design"],
        "final_mjcf_xml": best_iter["mjcf_xml"],
        "initial_metrics": iterations[0]["metrics"],
        "final_metrics": best_iter["metrics"],
        "initial_trace": iterations[0]["trace"],
        "final_trace": best_iter["trace"],
        "iterations": iterations,
        "naive_baseline": naive_baseline,
        "selected_iteration": best_iter["iteration"],
        "verification_scope": best_iter.get("verification_scope", sim_res.get("verification_scope", {})),
        "search_status": "passing_candidate_found" if best_iter["check"]["passed"] else "bounded_search_without_passing_candidate",
    }

    emit(
        "loop_completed",
        {
            "passed": result_bundle["passed"],
            "iterations_count": len(iterations),
            "final_design_id": best_iter["design"].get("design_id"),
            "initial_ee_rmse_mm": iterations[0]["metrics"]["ee_rmse_mm"],
            "final_ee_rmse_mm": best_iter["metrics"]["ee_rmse_mm"],
            "final_mass_kg": best_iter["metrics"]["total_mass_kg"],
            "final_cost_usd": best_iter["metrics"]["total_cost_usd"],
        },
    )
    return result_bundle
