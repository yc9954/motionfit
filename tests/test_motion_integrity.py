"""Regressions for measured input integrity and task-space verification."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend import gemini
from backend.kinematics import forward_kinematics, solve_trajectory_ik
from backend.motion import generate_preset_landmarks, landmarks_to_target
from backend.sim import compute_joint_derivatives, simulate_design
from backend.validate import validate_proposal

DEMO_DIR = Path(__file__).resolve().parents[1] / "backend" / "data" / "demo"


def design():
    return json.loads((DEMO_DIR / "design.json").read_text())


def test_trajectory_ik_tracks_over_shoulder_without_yaw_flip():
    links = design()["links"]
    expected = np.array([[0.0, a, -0.2, -0.5] for a in np.linspace(1.5, 2.0, 51)])
    positions = np.array([forward_kinematics(q, links)["ee"] for q in expected])
    result = solve_trajectory_ik(positions, expected[:, 1:].sum(axis=1), links)
    recovered = np.array([forward_kinematics(q, links)["ee"] for q in result["q_des"]])
    assert result["all_reachable"] and result["no_limit_violation"]
    assert np.max(np.linalg.norm(recovered - positions, axis=1)) < 0.0005
    assert np.max(np.abs(np.diff(result["q_des"], axis=0))) < 0.05


def test_orientation_constraint_changes_the_actual_target():
    links = design()["links"]
    measured = generate_preset_landmarks("pour", links=links)
    free = landmarks_to_target(measured, links, orientation_mode="free")
    level = landmarks_to_target(measured, links, orientation_mode="keep_level")
    upright = landmarks_to_target(measured, links, orientation_mode="keep_upright")
    assert np.max(np.abs(free["phi"])) > 0.2
    assert np.max(np.abs(level["phi"])) == 0
    assert np.allclose(upright["phi"], np.pi / 2, atol=1e-5)
    assert free["q_des"] != level["q_des"]


@pytest.mark.parametrize("fault", ["timestamp", "nan", "visibility", "missing_hip", "object_coordinates"])
def test_bad_measured_landmarks_are_rejected(fault):
    links = design()["links"]
    measured = generate_preset_landmarks("pour", links=links)
    if fault == "timestamp":
        measured["frames"][1]["t"] = measured["frames"][0]["t"]
    elif fault == "nan":
        measured["frames"][0]["wrist"][0] = float("nan")
    elif fault == "visibility":
        for frame in measured["frames"]:
            frame["visibility"] = 0.1
    elif fault == "missing_hip":
        del measured["frames"][0]["hip"]
    else:
        measured["frames"][0]["wrist"] = {"x": 0, "y": 0, "z": 0}
    with pytest.raises(ValueError):
        landmarks_to_target(measured, links)


def test_nonuniform_simulation_uses_exact_sample_times_and_original_target():
    current = design()
    pos = forward_kinematics([0, 0.8, -0.5, -0.3], current["links"])["ee"].tolist()
    target = {"t": [2.0, 2.007, 2.019], "p_ee": [pos, pos, pos], "phi": [0, 0, 0]}
    result = simulate_design(current, target)
    assert result["simulation_scope"]["simulated_duration_s"] == pytest.approx(0.019)
    assert result["trace"]["t"] == target["t"]
    for q, recorded in zip(result["trace"]["q_act"], result["trace"]["ee_act"]):
        assert np.linalg.norm(forward_kinematics(q, current["links"])["ee"] - recorded) < 0.0002
    unreachable = {"t": [0, 0.01, 0.02], "p_ee": [[2, 0, 0.2]] * 3, "phi": [0] * 3}
    failed = simulate_design(current, unreachable)
    assert failed["check"]["passed"] is False
    assert failed["metrics"]["ee_rmse_mm"] > 1000
    assert failed["trace"]["ee_des"] == unreachable["p_ee"]
    assert failed["trace"]["ee_ik"] != failed["trace"]["ee_des"]


@pytest.mark.parametrize("times", [[0, 0, 0.1], [0, -1, 1], [0, float("nan"), 1]])
def test_invalid_time_grid_rejected(times):
    with pytest.raises(ValueError):
        simulate_design(design(), {"t": times, "p_ee": [[0.4, 0, 0.3]] * 3, "phi": [0] * 3})
    with pytest.raises(ValueError):
        compute_joint_derivatives(np.array(times), np.zeros((3, 4)))


def test_catalog_extraction_never_substitutes_demo(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "_get_genai_client", lambda: None)
    with pytest.raises(ValueError, match="no catalog was substituted"):
        gemini.extract_catalog_from_pdfs([tmp_path / "user.pdf"])


def test_offline_intent_does_not_invent_task_or_payload(monkeypatch):
    monkeypatch.setattr(gemini, "_get_genai_client", lambda: None)
    result = gemini.analyze_motion_intent({"duration_s": 2.5})
    assert result["source"] == "measured_kinematics_only"
    assert result["phases"][0]["t_end"] == 2.5
    assert result["orientation_constraint"] == "free"
    assert "0.40" not in json.dumps(result)
    assert "wipe" not in result["task_summary"]


def test_catalog_citation_must_match_exact_part_field_page():
    current = design()
    catalog = json.loads((DEMO_DIR / "catalog.json").read_text())
    part = next(p for p in catalog["parts"] if p["part_id"] == "XM540-W270")
    src = part["stall_torque_nm"]["src"]
    proposal = {"changes": [{"target": "J2.actuator_id", "from": "XM430-W350", "to": "XM540-W270",
                             "rationale": "Use a candidate from the catalog.",
                             "citations": [{"doc": src["doc"], "page": src["page"] + 1, "field": "stall_torque_nm"}]}]}
    result = validate_proposal(proposal, current, catalog)
    assert not result["valid"]
    assert any("citation" in err or "cites" in err for err in result["errors"])


def test_offline_catalog_search_uses_custom_part_ids():
    current = design()
    catalog = json.loads((DEMO_DIR / "catalog.json").read_text())
    # Rename every part; selection must use measured quantities, not known IDs.
    rename = {p["part_id"]: f"custom-{i}" for i, p in enumerate(catalog["parts"])}
    for p in catalog["parts"]:
        p["part_id"] = rename[p["part_id"]]
    for cfg in current["joints"].values():
        cfg["actuator_id"] = rename[cfg["actuator_id"]]
    metrics = {"joints": {j: {"peak_req_nm": 0.1, "rms_req_nm": 0.05, "peak_speed_rad_s": 0.1,
                              "peak_ratio": 0.1, "rms_ratio": 0.1, "speed_ratio": 0.1} for j in current["joints"]}}
    metrics["joints"]["J3"].update(peak_req_nm=5, rms_req_nm=2, peak_ratio=2, rms_ratio=2)
    proposal = gemini._physics_guided_proposal(current, {"passed": False}, metrics, catalog)
    assert proposal["changes"]
    assert all(change["target"].startswith("J3.") for change in proposal["changes"])
    assert all(change["to"] is None or change["to"].startswith("custom-") for change in proposal["changes"])
    assert "wipe" not in proposal["diagnosis"]


def test_catalog_normalization_is_idempotent_and_preserves_input():
    from backend.catalog_normalize import normalize_catalog_part
    raw = {"part_id": "measured", "type": "actuator",
           "stall_torque_nm": {"value": 10, "unit": "kgf·cm", "src": {"doc": "own.pdf", "page": 1}},
           "continuous_torque_nm": {"value": None},
           "no_load_speed_rad_s": {"value": 30, "unit": "rpm", "src": {"doc": "own.pdf", "page": 1}},
           "mass_kg": {"value": 120, "unit": "g", "src": {"doc": "own.pdf", "page": 1}}}
    original = copy.deepcopy(raw)
    once = normalize_catalog_part(raw)
    twice = normalize_catalog_part(once)
    assert raw == original
    for field in ("stall_torque_nm", "continuous_torque_nm", "no_load_speed_rad_s", "mass_kg"):
        assert once[field]["value"] == twice[field]["value"]
    assert once["no_load_speed_rad_s"]["unit"] == "rad/s"
    assert once["mass_kg"]["unit"] == "kg"


def test_interactions_parser_accepts_sdk_and_dictionary_steps():
    response = {"steps": [{"type": "thought", "summary": [{"type": "text", "text": "Public summary"}]},
                          {"type": "model_output", "content": [{"type": "text", "text": '{"changes": []}'}]}]}
    assert gemini._extract_thought_and_text(response) == ("Public summary", '{"changes": []}')
    sdk_response = SimpleNamespace(steps=[SimpleNamespace(type="model_output", content=[SimpleNamespace(type="text", text="actual result")])])
    assert gemini._extract_thought_and_text(sdk_response)[1] == "actual result"


def test_pdf_extraction_checks_field_page(monkeypatch):
    part = {"part_id": "MY-MOTOR", "type": "actuator",
            **{field: {"value": 1, "src": {"doc": "xm540-w270.pdf", "page": 999}} for field in ("stall_torque_nm", "no_load_speed_rad_s", "mass_kg")}}
    client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=json.dumps({"parts": [part]}))))
    monkeypatch.setattr(gemini, "_get_genai_client", lambda: client)
    with pytest.raises(ValueError, match="existing page"):
        gemini.extract_catalog_from_pdfs([DEMO_DIR / "datasheets" / "xm540-w270.pdf"])


def test_report_uses_selected_iteration_and_truthful_failure_status():
    from backend.report import render_solution_report
    current = design()
    pos = forward_kinematics([0, 0.8, -0.5, -0.3], current["links"])["ee"].tolist()
    result = simulate_design(current, {"t": [0, 0.01, 0.02], "p_ee": [pos] * 3, "phi": [0] * 3})
    selected = {"design_id": "selected-result", "metrics": result["metrics"], "check": {"passed": True}, "trace": result["trace"], "proposal": None}
    later = copy.deepcopy(selected)
    later["design_id"] = "later-rejected"
    later["check"]["passed"] = False
    later["metrics"]["ee_rmse_mm"] = 9999
    current["design_id"] = "selected-result"
    catalog = json.loads((DEMO_DIR / "catalog.json").read_text())
    bundle = {"iterations": [selected, later], "passed": True, "final_metrics": selected["metrics"],
              "initial_trace": result["trace"], "final_trace": result["trace"]}
    summary = {"duration_s": 0.02, "peak_cart_speed_m_s": 0, "bbox_min": pos, "bbox_max": pos}
    html = render_solution_report("test", current, current, catalog, {}, summary, bundle)
    assert "Selected design selected-result passed verification" in html
    assert f"target tracking RMSE {selected['metrics']['ee_rmse_mm']:.3f} mm" in html
    assert "not manufacturer original" in html
    selected["check"]["passed"] = False
    bundle["iterations"] = [selected]
    bundle["passed"] = False
    summary = {"duration_s": 0.02, "peak_cart_speed_m_s": 0, "bbox_min": pos, "bbox_max": pos}
    html = render_solution_report("test", current, current, catalog, {}, summary, bundle)
    assert "Selected design selected-result failed verification" in html
    assert "Selected design selected-result passed verification" not in html


def test_mediapipe_world_is_retargeted_once_using_the_torso_frame():
    links = design()["links"]
    measured = generate_preset_landmarks("pour", links=links)
    original = landmarks_to_target(measured, links)
    camera_basis = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)
    for frame in measured["frames"]:
        for name in ("shoulder", "elbow", "wrist", "index", "hip"):
            frame[name] = (np.array(frame[name]) @ camera_basis.T).tolist()
    measured["coordinate_system"] = "mediapipe_world_m"
    measured["source"] = "mediapipe_pose_world"
    retargeted = landmarks_to_target(measured, links)
    positions = np.asarray(retargeted["p_ee"])
    expected = np.asarray(original["p_ee"])
    expected[:, 0] += retargeted["retargeting"]["forward_translation_m"]
    assert np.max(np.abs(positions - expected)) < 0.00003
    assert np.min(positions[:, 0]) > 0.03
    assert retargeted["source"] == "mediapipe_pose_world"
    assert np.allclose(retargeted["phi"], original["phi"])


def test_constrained_tool_orientation_is_verified_on_actual_simulation():
    from backend.checker import evaluate_design
    current = design()
    current["payload_kg"] = 5.0
    pos = forward_kinematics([0, 0.3, -0.15, -0.15], current["links"])["ee"].tolist()
    times = np.linspace(0, 0.5, 51).tolist()
    result = simulate_design(current, {"t": times, "p_ee": [pos] * len(times), "phi": [0] * len(times), "orientation_mode": "keep_level"})
    trace = result["trace"]
    assert np.allclose(trace["phi_des"], 0)
    assert np.allclose(trace["phi_act"], np.sum(np.asarray(trace["q_act"])[:, 1:], axis=1), atol=0.0002)
    assert result["metrics"]["tool_pitch_max_error_deg"] > 5
    assert result["check"]["checks"]["tool_orientation"] is False
    # Independently isolate orientation from all other checks: a passing point
    # tracking score must not hide a violated upright/level requirement.
    metrics = copy.deepcopy(result["metrics"])
    metrics["ee_rmse_mm"] = 0
    metrics["total_cost_usd"] = metrics["total_mass_kg"] = 0
    for jm in metrics["joints"].values():
        jm.update(peak_ratio=0, rms_ratio=0, speed_ratio=0)
    judged = evaluate_design(metrics, current, ik_result={"unclamped_fraction": 1})
    assert not judged["passed"]
    assert [v["check"] for v in judged["violations"]] == ["tool_orientation"]
