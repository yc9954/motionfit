"""FastAPI application for MotionFit (Section 7)."""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import google.genai as genai
import mujoco
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.catalog_normalize import load_prices_csv, normalize_catalog
from backend.gemini import ARGON_MODEL, analyze_motion_intent, extract_catalog_from_pdfs
from backend.loop import run_design_loop
from backend.model_builder import build_mjcf, load_materials
from backend.motion import generate_preset_landmarks, process_landmarks_to_target, summarize_motion
from backend.report import render_solution_report
from backend.storage import (
    build_export_zip,
    create_project,
    ensure_demo_project,
    get_project_dir,
    get_project_state,
    load_json,
    save_json,
    save_text,
)


app = FastAPI(
    title="MotionFit API",
    description="Hardware Verification & Actuator Selection Powered by Gemini 4 Argon + MuJoCo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for SSE run events
RUN_EVENTS: Dict[str, List[Dict[str, Any]]] = {}
RUN_STATUS: Dict[str, str] = {}


class CreateProjectRequest(BaseModel):
    use_demo: bool = True
    design: Optional[Dict[str, Any]] = None


class MotionRequest(BaseModel):
    preset: Optional[str] = None
    landmarks: Optional[Dict[str, Any]] = None
    orientation_constraint: Optional[str] = None


class RunLoopRequest(BaseModel):
    max_iterations: int = Field(default=4, ge=1, le=5)
    enable_post_pass_optimization: bool = True
    include_naive_baseline: bool = True


class ApproveRequest(BaseModel):
    engineer_name: str = Field(..., min_length=1)
    notes: Optional[str] = ""


# Ensure the demo project is seeded and pre-warmed
ensure_demo_project()


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Return service health, MuJoCo version, google-genai version, and active model."""
    return {
        "status": "ok",
        "service": "motionfit",
        "mujoco_version": mujoco.__version__,
        "genai_version": getattr(genai, "__version__", "2.6.0"),
        "argon_model": ARGON_MODEL,
    }


@app.post("/api/projects")
async def api_create_project(req: CreateProjectRequest) -> Dict[str, Any]:
    """Create a project (or return seeded demo project)."""
    project_id = create_project(use_demo=req.use_demo, design=req.design)
    return get_project_state(project_id)


@app.get("/api/projects/{project_id}")
async def api_get_project(project_id: str) -> Dict[str, Any]:
    """Fetch complete project state."""
    try:
        return get_project_state(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/projects/{project_id}/design")
async def api_update_design(project_id: str, design: Dict[str, Any]) -> Dict[str, Any]:
    """Update design.json for a project."""
    try:
        save_json(project_id, "design.json", design)
        return {"status": "updated", "design": design}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/catalog/extract")
async def api_extract_catalog(project_id: str, force_live: bool = False) -> Dict[str, Any]:
    """Return cached catalog.json or extract from PDFs via Argon (with deterministic normalization)."""
    try:
        proj_dir = get_project_dir(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cat_path = proj_dir / "catalog.json"
    if cat_path.exists() and not force_live:
        catalog = json.loads(cat_path.read_text())
        return {"status": "cached", "catalog": catalog}

    ds_dir = proj_dir / "datasheets"
    pdf_paths = sorted(ds_dir.glob("*.pdf")) if ds_dir.exists() else []
    catalog = extract_catalog_from_pdfs(
        pdf_paths=pdf_paths,
        prices_path=proj_dir / "prices.csv",
        cached_catalog_path=None if force_live else cat_path,
    )
    save_json(project_id, "catalog.json", catalog)
    return {"status": "extracted", "catalog": catalog}


@app.put("/api/projects/{project_id}/catalog")
async def api_update_catalog(project_id: str, catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Save engineer-reviewed catalog.json (HITL step 1)."""
    try:
        for p in catalog.get("parts", []):
            p["user_confirmed"] = True
        save_json(project_id, "catalog.json", catalog)
        return {"status": "confirmed", "catalog": catalog}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/motion")
async def api_process_motion(project_id: str, req: MotionRequest) -> Dict[str, Any]:
    """Process uploaded pose landmarks or switch to a built-in motion preset (wipe, pour, pick_and_place)."""
    try:
        proj_dir = get_project_dir(project_id)
        design = load_json(project_id, "design.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if req.landmarks is not None:
        landmarks = req.landmarks
        preset_name = landmarks.get("preset", "custom")
    else:
        preset_name = req.preset or "water_bottle"
        if preset_name in ("water_bottle", "wipe") and (proj_dir / "landmarks.json").exists():
            landmarks = load_json(project_id, "landmarks.json")
        else:
            landmarks = generate_preset_landmarks(preset_name=preset_name, links=design.get("links"))

    save_json(project_id, "landmarks.json", landmarks)

    motion_summary = summarize_motion(landmarks)
    if preset_name in ("water_bottle", "wipe") and (proj_dir / "intent.json").exists() and req.landmarks is None and req.orientation_constraint is None:
        intent = load_json(project_id, "intent.json")
    else:
        intent = analyze_motion_intent(landmarks_summary=motion_summary)
        if req.orientation_constraint:
            intent["orientation_constraint"] = req.orientation_constraint

    target = process_landmarks_to_target(
        landmarks_data=landmarks,
        design=design,
        orientation_mode=intent.get("orientation_constraint", "free"),
    )
    save_json(project_id, "intent.json", intent)
    save_json(project_id, "target.json", target)

    return {
        "status": "processed",
        "preset": preset_name,
        "motion_summary": motion_summary,
        "intent": intent,
        "target": target,
    }


@app.post("/api/projects/{project_id}/run")
async def api_run_loop(project_id: str, req: Optional[RunLoopRequest] = None) -> Dict[str, Any]:
    """Execute the Argon hardware verification and redesign loop, recording SSE events."""
    if req is None:
        req = RunLoopRequest()
    try:
        design = load_json(project_id, "design.json")
        catalog = load_json(project_id, "catalog.json")
        target = load_json(project_id, "target.json")
        intent = load_json(project_id, "intent.json")
        landmarks = load_json(project_id, "landmarks.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    materials = load_materials()
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    events: List[Dict[str, Any]] = []
    RUN_EVENTS[run_id] = events
    RUN_STATUS[run_id] = "running"

    def _on_event(evt: Dict[str, Any]) -> None:
        evt_with_ts = {
            **evt,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        events.append(evt_with_ts)

    # Run the physics + Argon loop in a worker thread so the event loop stays responsive
    loop_result = await asyncio.to_thread(
        run_design_loop,
        initial_design=design,
        catalog=catalog,
        materials=materials,
        target=target,
        intent=intent,
        max_iterations=req.max_iterations,
        optimize_after_pass=req.enable_post_pass_optimization,
        include_naive_baseline=req.include_naive_baseline,
        on_event=_on_event,
    )

    final_design = loop_result["final_design"]
    final_xml = build_mjcf(final_design, catalog, materials)
    motion_summary = summarize_motion(landmarks)

    # Reset prior approval when a new loop is run
    approval = {
        "approved": False,
        "engineer_name": None,
        "notes": None,
        "approved_at": None,
    }

    report_html = render_solution_report(
        project_id=project_id,
        initial_design=design,
        final_design=final_design,
        catalog=catalog,
        intent=intent,
        motion_summary=motion_summary,
        loop_result=loop_result,
        approval=approval,
    )

    save_json(project_id, "loop_result.json", loop_result)
    save_json(project_id, "design_final.json", final_design)
    save_json(project_id, "iterations.json", loop_result["iterations"])
    save_json(project_id, "approval.json", approval)
    save_text(project_id, "arm_final.xml", final_xml, content_type="application/xml")
    save_text(project_id, "report.html", report_html, content_type="text/html")

    _on_event(
        {
            "type": "complete",
            "passed": loop_result["passed"],
            "final_design_id": final_design["design_id"],
            "iterations_count": len(loop_result["iterations"]),
        }
    )
    RUN_STATUS[run_id] = "completed"

    return {
        "run_id": run_id,
        "status": "completed",
        "passed": loop_result["passed"],
        "loop_result": loop_result,
        "events": events,
    }


@app.get("/api/runs/{run_id}/events")
async def api_stream_run_events(run_id: str) -> StreamingResponse:
    """Stream Server-Sent Events (SSE) for a design loop run."""
    if run_id not in RUN_EVENTS:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    async def _event_generator() -> AsyncGenerator[str, None]:
        idx = 0
        while True:
            evts = RUN_EVENTS.get(run_id, [])
            while idx < len(evts):
                payload = json.dumps(evts[idx])
                yield f"data: {payload}\n\n"
                idx += 1
            if RUN_STATUS.get(run_id) in ("completed", "failed"):
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.post("/api/projects/{project_id}/approve")
async def api_approve_design(project_id: str, req: ApproveRequest) -> Dict[str, Any]:
    """Record Human-in-the-Loop engineer sign-off and unlock export.zip."""
    try:
        design = load_json(project_id, "design.json")
        final_design = load_json(project_id, "design_final.json")
        catalog = load_json(project_id, "catalog.json")
        intent = load_json(project_id, "intent.json")
        landmarks = load_json(project_id, "landmarks.json")
        loop_result = load_json(project_id, "loop_result.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Run the verification loop before approving: {exc}") from exc

    approval = {
        "approved": True,
        "engineer_name": req.engineer_name.strip(),
        "notes": (req.notes or "").strip(),
        "approved_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    save_json(project_id, "approval.json", approval)

    motion_summary = summarize_motion(landmarks)
    report_html = render_solution_report(
        project_id=project_id,
        initial_design=design,
        final_design=final_design,
        catalog=catalog,
        intent=intent,
        motion_summary=motion_summary,
        loop_result=loop_result,
        approval=approval,
    )
    save_text(project_id, "report.html", report_html, content_type="text/html")

    return {"status": "approved", "approval": approval}


@app.get("/api/projects/{project_id}/report", response_class=HTMLResponse)
async def api_get_report(project_id: str) -> HTMLResponse:
    """Return the rendered HTML solution report."""
    try:
        proj_dir = get_project_dir(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    report_path = proj_dir / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report has not been generated yet. Run the loop first.")
    return HTMLResponse(content=report_path.read_text())


@app.get("/api/projects/{project_id}/export.zip")
async def api_export_zip(project_id: str) -> Response:
    """Download the verified design bundle (gated on engineer approval)."""
    try:
        proj_dir = get_project_dir(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    approval_path = proj_dir / "approval.json"
    if not approval_path.exists():
        raise HTTPException(status_code=403, detail="Engineer approval is required before exporting the final design.")
    approval = json.loads(approval_path.read_text())
    if not approval.get("approved"):
        raise HTTPException(status_code=403, detail="Engineer approval is required before exporting the final design.")

    zip_bytes = build_export_zip(project_id)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="motionfit_{project_id}_verified_bundle.zip"'},
    )


@app.get("/api/projects/{project_id}/datasheets/{doc_name}")
async def api_get_datasheet(project_id: str, doc_name: str) -> FileResponse:
    """Serve a datasheet PDF so citation links open the exact PDF page."""
    try:
        proj_dir = get_project_dir(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    safe_name = Path(doc_name).name
    pdf_path = proj_dir / "datasheets" / safe_name
    if not pdf_path.exists():
        # Fallback to demo datasheets
        demo_pdf = ensure_demo_project() / "datasheets" / safe_name
        if demo_pdf.exists():
            pdf_path = demo_pdf
        else:
            raise HTTPException(status_code=404, detail=f"Datasheet '{safe_name}' not found")
    return FileResponse(path=str(pdf_path), media_type="application/pdf", filename=safe_name)


# Mount built frontend static directory if it exists
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
