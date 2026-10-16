"""Regression coverage for measured-input, asynchronous, versioned project workflow."""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import main, storage
from backend.motion import generate_preset_landmarks
from backend.task_planner import plan_task


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTIONFIT_OFFLINE", "1")
    monkeypatch.delenv("MOTIONFIT_GCS_BUCKET", raising=False)
    monkeypatch.setattr(storage, "PROJECTS_DIR", tmp_path / "projects")
    main.PROJECT_LOCKS.clear()
    main.RUN_RECORDS.clear()
    main.RUN_STATUS.clear()
    main.RUN_EVENTS.clear()
    with TestClient(main.app) as client:
        yield client
    for future in list(main.RUN_FUTURES.values()):
        future.result(timeout=15)
    main.RUN_FUTURES.clear()


def create_ready(api, *, confirm=True):
    project = api.post("/api/projects", json={"use_demo": False}).json()
    project_id = project["project_id"]
    response = api.post(f"/api/projects/{project_id}/task", json={"prompt": "Move this measured object", "constraints": {"payload_kg": .5, "max_cost_usd": 3000, "max_total_mass_kg": 5, "min_reach_m": .5}})
    assert response.status_code == 200, response.text
    if confirm:
        assert api.put(f"/api/projects/{project_id}/catalog", json=project["catalog"]).status_code == 200
    landmarks = generate_preset_landmarks()
    landmarks["source"] = "test_measured_landmarks"
    response = api.post(f"/api/projects/{project_id}/motion", json={"landmarks": landmarks})
    assert response.status_code == 200, response.text
    return project_id


def wait_run(api, run_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = api.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(.01)
    pytest.fail("Run did not finish")


def stub_passing_loop(monkeypatch):
    fixture = json.loads((storage.DEMO_DIR / "loop_result.json").read_text())
    def run(**kwargs):
        result = copy.deepcopy(fixture)
        result["initial_design"] = kwargs["initial_design"]
        result["final_design"] = copy.deepcopy(kwargs["initial_design"])
        result["final_design"]["design_id"] += "-verified"
        kwargs["on_event"]({"event": "iteration_simulated", "iteration": 0, "passed": True})
        return result
    monkeypatch.setattr(main, "run_design_loop", run)
    monkeypatch.setattr(main, "render_solution_report", lambda **kwargs: "<html>verified</html>")


def test_new_projects_are_isolated_without_stale_demo_results(api):
    first = api.post("/api/projects", json={}).json()
    second = api.post("/api/projects", json={}).json()
    assert first["project_id"] != second["project_id"] != "demo"
    assert first["landmarks"] is None and first["loop_result"] is None
    assert first["target"] is None and first["approval"] is None
    assert first["catalog"]["confirmed"] is False
    assert first["catalog"]["provenance"]["manufacturer_originals"] is False
    assert all(not part["user_confirmed"] for part in first["catalog"]["parts"])
    with pytest.raises(ValueError):
        storage.get_project_dir("../outside")


def test_task_parser_requires_mass_and_preserves_explicit_requirements(api):
    task = plan_task("500ml 물병을 수평으로 옮겨줘")
    assert task["constraints"]["payload_kg"] is None
    assert task["missing_fields"] == ["payload_kg"]
    task = plan_task("하중 700g, 예산 1800달러, 총 무게 4kg, 도달거리 60cm")
    assert task["constraints"] == {"payload_kg": .7, "max_cost_usd": 1800, "max_total_mass_kg": 4, "min_reach_m": .6}
    with pytest.raises(ValueError):
        plan_task("Move", {"payload_kg": -1})


def test_motion_rejects_implicit_preset_and_noncanonical_landmarks(api):
    project_id = api.post("/api/projects", json={}).json()["project_id"]
    assert api.post(f"/api/projects/{project_id}/motion", json={}).status_code == 422
    data = generate_preset_landmarks()
    data["frames"][1]["wrist"] = {"x": .1, "y": .1, "z": .1}
    response = api.post(f"/api/projects/{project_id}/motion", json={"landmarks": data})
    assert response.status_code == 422
    assert "[x, y, z]" in response.json()["detail"]
    assert api.get(f"/api/projects/{project_id}").json()["landmarks"] is None


def test_run_requires_catalog_review(api):
    project_id = create_ready(api, confirm=False)
    response = api.post(f"/api/projects/{project_id}/run", json={"max_iterations": 1})
    assert response.status_code == 409
    assert "confirm" in response.text


def test_async_run_returns_promptly_locks_inputs_and_reports_failure(api, monkeypatch):
    project_id = create_ready(api)
    started, release = threading.Event(), threading.Event()
    def blocked_run(**kwargs):
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("deterministic test failure")
    monkeypatch.setattr(main, "run_design_loop", blocked_run)
    response = api.post(f"/api/projects/{project_id}/run", json={"max_iterations": 1})
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert started.wait(timeout=2)
    assert api.post(f"/api/projects/{project_id}/task", json={"prompt": "Another task"}).status_code == 409
    release.set()
    result = wait_run(api, response.json()["run_id"])
    assert result["status"] == "failed"
    assert "deterministic test failure" in result["error"]
    events = api.get(f"/api/runs/{result['run_id']}/events")
    assert '"event": "failed"' in events.text
    # Failure releases the project lock.
    assert api.post(f"/api/projects/{project_id}/task", json={"prompt": "Try again"}).status_code == 200
    assert api.post(f"/api/projects/{project_id}/approve", json={"engineer_name": "Engineer"}).status_code == 409


def test_acceptance_approval_export_bind_to_verified_input_revision(api, monkeypatch):
    stub_passing_loop(monkeypatch)
    project_id = create_ready(api)
    run = api.post(f"/api/projects/{project_id}/run", json={"max_iterations": 1}).json()
    result = wait_run(api, run["run_id"])
    assert result["status"] == "completed", result
    assert result["loop_result"]["input_fingerprint"] == run["fingerprint"]
    assert api.post(f"/api/projects/{project_id}/approve", json={"engineer_name": "Engineer"}).status_code == 409
    assert api.post(f"/api/projects/{project_id}/accept", json={"run_id": "old-run"}).status_code == 409
    assert api.post(f"/api/projects/{project_id}/accept", json={"run_id": run["run_id"]}).status_code == 200
    assert api.post(f"/api/projects/{project_id}/approve", json={"engineer_name": "Engineer"}).status_code == 200
    assert api.get(f"/api/projects/{project_id}/export.zip").status_code == 200
    # Requirements update removes candidate/report and invalidates the approval.
    response = api.post(f"/api/projects/{project_id}/task", json={"prompt": "Increase payload", "constraints": {"payload_kg": .8}})
    assert response.status_code == 200
    project = api.get(f"/api/projects/{project_id}").json()
    assert project["loop_result"] is None and project["accepted"] is False
    assert project["approval"]["approved"] is False
    assert api.get(f"/api/projects/{project_id}/export.zip").status_code == 403
    assert api.get(f"/api/runs/{run['run_id']}").json()["stale"] is True


def test_fingerprint_detects_out_of_band_change_and_restart_status(api, monkeypatch):
    stub_passing_loop(monkeypatch)
    project_id = create_ready(api)
    run = api.post(f"/api/projects/{project_id}/run", json={"max_iterations": 1}).json()
    wait_run(api, run["run_id"])
    design = storage.load_json(project_id, "design.json")
    design["payload_kg"] = 1.1
    storage.save_json(project_id, "design.json", design)
    assert api.post(f"/api/projects/{project_id}/accept", json={"run_id": run["run_id"]}).status_code == 409
    # A completed run can be inspected from persisted state after memory loss.
    main.RUN_RECORDS.clear()
    assert api.get(f"/api/runs/{run['run_id']}").json()["status"] == "completed"


def test_configured_gcs_failure_is_visible_and_never_partial_local_write(api, monkeypatch):
    project = api.post("/api/projects", json={}).json()
    path = storage.get_project_dir(project["project_id"]) / "design.json"
    before = path.read_bytes()
    monkeypatch.setattr(storage, "_maybe_sync_to_gcs", lambda *args, **kwargs: (_ for _ in ()).throw(storage.StorageSyncError("remote failed")))
    with pytest.raises(storage.StorageSyncError):
        storage.save_json(project["project_id"], "design.json", {"corrupt": True})
    assert path.read_bytes() == before


def test_catalog_rejects_invalid_unused_candidate_and_missing_citation_page(api):
    project = api.post("/api/projects", json={}).json()
    project_id = project["project_id"]
    invalid = copy.deepcopy(project["catalog"])
    invalid["parts"][-1]["mass_kg"]["value"] = None
    assert api.put(f"/api/projects/{project_id}/catalog", json=invalid).status_code == 422
    invalid = copy.deepcopy(project["catalog"])
    invalid["parts"][0]["stall_torque_nm"]["src"]["page"] = 999
    assert api.put(f"/api/projects/{project_id}/catalog", json=invalid).status_code == 422
    assert api.get(f"/api/projects/{project_id}").json()["catalog"]["confirmed"] is False


def test_catalog_rebases_missing_initial_parts_to_labelled_unverified_baseline(api):
    project = api.post("/api/projects", json={}).json()
    project_id = project["project_id"]
    catalog = copy.deepcopy(project["catalog"])
    catalog["parts"] = [part for part in catalog["parts"] if part["part_id"] == "XM540-W270"]
    response = api.put(f"/api/projects/{project_id}/catalog", json=catalog)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["baseline_changes"]) == 4
    assert "unverified starting design" in payload["baseline_notice"]
    assert all(config["actuator_id"] == "XM540-W270" for config in payload["design"]["joints"].values())
    assert api.get(f"/api/projects/{project_id}").json()["loop_result"] is None


def test_invalid_initial_design_is_not_persisted_and_upload_requires_reextraction(api):
    response = api.post("/api/projects", json={"design": {"payload_kg": -1}})
    assert response.status_code == 422
    assert not storage.PROJECTS_DIR.exists()
    project = api.post("/api/projects", json={}).json()
    project_id = project["project_id"]
    response = api.post(f"/api/projects/{project_id}/catalog/files", files=[("files", ("review.pdf", (storage.DEMO_DIR / "datasheets" / "xm540-w270.pdf").read_bytes(), "application/pdf"))])
    assert response.status_code == 200
    # An old JSON body cannot bypass the persisted source-change flag.
    assert api.put(f"/api/projects/{project_id}/catalog", json=project["catalog"]).status_code == 422
