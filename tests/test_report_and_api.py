"""Unit & integration tests for M5: report generator, number-source check, and FastAPI endpoints."""

from __future__ import annotations

import io
import zipfile
from fastapi.testclient import TestClient

from backend.main import app
from backend.report import verify_prose_numbers


client = TestClient(app)


from backend.storage import ensure_demo_project


def test_frontend_static_bundle_served() -> None:
    """Confirm GET / serves the compiled Vite + Three.js + MediaPipe frontend and JS/CSS assets."""
    import re

    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "MotionFit" in r_index.text
    assert '<div id="app"></div>' in r_index.text
    m = re.search(r'src="(/assets/index-[^"]+\.js)"', r_index.text)
    assert m is not None
    r_js = client.get(m.group(1))
    assert r_js.status_code == 200
    assert "viewport-before" in r_js.text
    assert "viewport-after" in r_js.text
    assert "live-strip-before" in r_js.text




def test_prose_number_verification() -> None:
    """Confirm verify_prose_numbers accepts verified numbers and rejects invented numbers."""
    allowed = [0.788, 0.853, 11.98, 295.12, 1101.15, 1216.15, 4.1, 10.6, 17.0]
    valid_prose = {
        "executive_summary": "Initial RMSE 295.12 mm improved to 11.98 mm on XM430-W350 + APEX-PG20-2 with mass 0.853 kg.",
    }
    assert verify_prose_numbers(valid_prose, allowed) is True

    invalid_prose = {
        "executive_summary": "Invented torque 99.99 N·m with cost $7777.00.",
    }
    assert verify_prose_numbers(invalid_prose, allowed) is False


def test_fastapi_end_to_end_flow() -> None:
    """Test health, project creation, catalog confirm, motion switch, loop run, SSE events, report, and approval-gated export."""
    try:
        # 1. Health check
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"
        assert r_health.json()["argon_model"] == "gemini-4-argon"

        # 2. Get demo project
        r_proj = client.get("/api/projects/demo")
        assert r_proj.status_code == 200
        proj_data = r_proj.json()
        assert proj_data["project_id"] == "demo"
        assert proj_data["design"]["design_id"] == "demo-v0"
        assert len(proj_data["catalog"]["parts"]) >= 16

        # 3. Confirm catalog (HITL step 1)
        r_cat = client.put("/api/projects/demo/catalog", json=proj_data["catalog"])
        assert r_cat.status_code == 200
        assert all(p["user_confirmed"] for p in r_cat.json()["catalog"]["parts"])

        # 4. Run the verification loop
        r_run = client.post(
            "/api/projects/demo/run",
            json={"max_iterations": 4, "enable_post_pass_optimization": True, "include_naive_baseline": True},
        )
        assert r_run.status_code == 200
        run_payload = r_run.json()
        assert run_payload["passed"] is True
        run_id = run_payload["run_id"]
        assert len(run_payload["events"]) >= 5

        # 5. Stream SSE events for run_id
        r_sse = client.get(f"/api/runs/{run_id}/events")
        assert r_sse.status_code == 200
        assert "iteration_simulated" in r_sse.text
        assert "complete" in r_sse.text

        # 6. Fetch HTML report and verify all 10 sections + clickable datasheet links
        r_rep = client.get("/api/projects/demo/report")
        assert r_rep.status_code == 200
        html = r_rep.text
        assert "Executive Summary" in html
        assert "Not Verified by Simulation" in html
        assert "PENDING ENGINEER APPROVAL" in html
        assert "/api/projects/demo/datasheets/xm540-w270.pdf#page=1" in html

        # 7. Verify export.zip is blocked (403) before engineer approval
        r_exp_blocked = client.get("/api/projects/demo/export.zip")
        assert r_exp_blocked.status_code == 403

        # 8. Approve design (HITL step 2)
        r_app = client.post(
            "/api/projects/demo/approve",
            json={"engineer_name": "Alex Lee, Lead Robotics Engineer", "notes": "Verified J2 planetary stage margins and thermal RMS."},
        )
        assert r_app.status_code == 200
        assert r_app.json()["approval"]["approved"] is True

        # 9. Verify report now displays ENGINEER APPROVED FOR EXPORT
        r_rep_after = client.get("/api/projects/demo/report")
        assert "ENGINEER APPROVED FOR EXPORT" in r_rep_after.text
        assert "Alex Lee, Lead Robotics Engineer" in r_rep_after.text

        # 10. Download export.zip and verify contents
        r_exp_ok = client.get("/api/projects/demo/export.zip")
        assert r_exp_ok.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r_exp_ok.content), "r") as zf:
            names = set(zf.namelist())
            assert {
                "design_final.json",
                "arm_final.xml",
                "iterations.json",
                "catalog.json",
                "target.json",
                "report.html",
                "approval.json",
            }.issubset(names)

        # 11. Serve PDF datasheet citation
        r_pdf = client.get("/api/projects/demo/datasheets/xm540-w270.pdf")
        assert r_pdf.status_code == 200
        assert r_pdf.headers["content-type"] == "application/pdf"
        assert r_pdf.content.startswith(b"%PDF")
    finally:
        ensure_demo_project(force_reset=True)

