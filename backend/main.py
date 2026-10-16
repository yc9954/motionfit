"""MotionFit API: user requirements -> measured motion -> verified, versioned design."""
from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import io
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import google.genai as genai
import mujoco
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.catalog_normalize import normalize_catalog
from backend.gemini import ARGON_MODEL, extract_catalog_from_pdfs
from backend.loop import run_design_loop
from backend.model_builder import build_mjcf, load_materials
from backend.motion import generate_preset_landmarks, process_landmarks_to_target, summarize_motion
from backend.report import render_solution_report
from backend import storage
from backend.storage import build_export_zip, create_project, ensure_demo_project, get_project_dir, get_project_state, input_fingerprint, invalidate_results, load_json, save_bytes, save_json, save_text
from backend.task_planner import apply_task_to_design, plan_task

app = FastAPI(title="MotionFit API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
RUN_EVENTS: Dict[str, List[Dict[str, Any]]] = {}
RUN_STATUS: Dict[str, str] = {}
RUN_RECORDS: Dict[str, Dict[str, Any]] = {}
RUN_FUTURES: Dict[str, Any] = {}
PROJECT_LOCKS: Dict[str, threading.Lock] = {}
LOCK_GUARD = threading.Lock()
RUN_GUARD = threading.RLock()
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="motionfit")
MAX_EVENTS = 256
MAX_CACHED_RUNS = 64


class CreateProjectRequest(BaseModel):
    use_demo: bool = False
    design: Optional[Dict[str, Any]] = None


class TaskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    constraints: Optional[Dict[str, Any]] = None
    orientation_constraint: Optional[str] = None


class MotionRequest(BaseModel):
    preset: Optional[str] = None
    landmarks: Optional[Dict[str, Any]] = None
    orientation_constraint: Optional[str] = None


class RunLoopRequest(BaseModel):
    max_iterations: int = Field(default=4, ge=1, le=5)
    enable_post_pass_optimization: bool = True
    include_naive_baseline: bool = False


class AcceptRequest(BaseModel):
    run_id: Optional[str] = None


class ApproveRequest(BaseModel):
    engineer_name: str = Field(min_length=1, max_length=200)
    notes: Optional[str] = Field(default="", max_length=4000)


@app.exception_handler(FileNotFoundError)
async def missing_file(_request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def invalid_input(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(storage.StorageSyncError)
async def storage_error(_request: Request, exc: storage.StorageSyncError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _project_lock(project_id: str) -> threading.Lock:
    get_project_dir(project_id)
    with LOCK_GUARD:
        return PROJECT_LOCKS.setdefault(project_id, threading.Lock())


@contextlib.contextmanager
def _editing(project_id: str):
    lock = _project_lock(project_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="This project is already being verified or updated. Wait for its run to finish.")
    try:
        yield
    finally:
        lock.release()


def _optional(project_id: str, filename: str, default: Any = None) -> Any:
    try:
        return load_json(project_id, filename)
    except FileNotFoundError:
        return default


def _validate_design(design: Dict[str, Any], catalog: Dict[str, Any]) -> None:
    json.dumps(design, allow_nan=False)
    try:
        if not isinstance(design["payload_kg"], (int, float)) or isinstance(design["payload_kg"], bool) or not 0 <= design["payload_kg"] <= 100:
            raise ValueError("payload_kg must be a finite number from 0 to 100")
        for name in ("base_height_m", "upper_arm_m", "forearm_m", "wrist_to_tool_m"):
            if not 0 < float(design["links"][name]) <= 5:
                raise ValueError(f"links.{name} must be positive and at most 5 m")
        for name in ("ee_rmse_mm", "peak_torque_ratio", "rms_torque_ratio", "peak_speed_ratio"):
            if not math.isfinite(float(design["thresholds"][name])) or float(design["thresholds"][name]) <= 0:
                raise ValueError(f"thresholds.{name} must be positive")
        for name in ("max_total_mass_kg", "max_cost_usd", "min_reach_m"):
            if not math.isfinite(float(design["constraints"][name])) or float(design["constraints"][name]) < 0:
                raise ValueError(f"constraints.{name} must be nonnegative")
        model = mujoco.MjModel.from_xml_string(build_mjcf(design, catalog, load_materials()))
        if model.nv != 4:
            raise ValueError("Only the 4-DOF arm model is supported")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete or invalid design: {exc}") from exc


def _validate_catalog(catalog: Dict[str, Any], project_id: str) -> None:
    json.dumps(catalog, allow_nan=False)
    parts = catalog.get("parts")
    if not isinstance(parts, list) or not parts or len(parts) > 500:
        raise ValueError("Catalog must contain 1 to 500 reviewed parts")
    ids = set()
    document_pages: Dict[str, int] = {}
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("part_id"), str) or not part["part_id"].strip() or part["part_id"] in ids:
            raise ValueError("Catalog part IDs must be nonempty and unique")
        ids.add(part["part_id"])
        kind = part.get("type")
        if kind not in ("actuator", "gearbox"):
            raise ValueError(f"Unsupported catalog part type: {kind}")
        fields = ("stall_torque_nm", "continuous_torque_nm", "no_load_speed_rad_s", "mass_kg", "price_usd") if kind == "actuator" else ("ratio", "efficiency", "rated_output_torque_nm", "mass_kg", "price_usd")
        for field in fields:
            spec = part.get(field)
            value = spec.get("value") if isinstance(spec, dict) else None
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (field != "price_usd" and value == 0):
                raise ValueError(f"{part['part_id']}.{field} requires a finite {'nonnegative' if field == 'price_usd' else 'positive'} reviewed SI value")
            if field != "price_usd":
                source = spec.get("src", {})
                doc, page = source.get("doc"), source.get("page")
                if not isinstance(doc, str) or Path(doc).name != doc or not doc.lower().endswith(".pdf") or not isinstance(page, int) or page < 1:
                    raise ValueError(f"{part['part_id']}.{field} requires an exact PDF filename and page citation")
                pdf_path = get_project_dir(project_id) / "datasheets" / doc.lower()
                if not pdf_path.exists():
                    raise ValueError(f"Catalog citation {doc} is missing from this project")
                if doc not in document_pages:
                    from pypdf import PdfReader
                    document_pages[doc] = len(PdfReader(pdf_path).pages)
                if page > document_pages[doc]:
                    raise ValueError(f"Catalog citation {doc} page {page} does not exist")
        if kind == "actuator" and part["continuous_torque_nm"]["value"] > part["stall_torque_nm"]["value"]:
            raise ValueError(f"{part['part_id']}: continuous torque exceeds stall torque")
        if kind == "gearbox" and (part["ratio"]["value"] < 1 or part["efficiency"]["value"] > 1):
            raise ValueError(f"{part['part_id']}: gearbox ratio must be >= 1 and efficiency <= 1")
    if not any(part.get("type") == "actuator" for part in parts):
        raise ValueError("Catalog requires at least one actuator")


def _validate_landmarks(landmarks: Dict[str, Any]) -> None:
    frames = landmarks.get("frames")
    if not isinstance(frames, list) or not 15 <= len(frames) <= 18000:
        raise ValueError("Motion must contain between 15 and 18000 measured frames")
    if landmarks.get("source") in ("synthetic_fallback", "fallback", "synthetic"):
        raise ValueError("Synthetic fallback landmarks cannot be submitted as measured video")
    previous = -math.inf
    for frame in frames:
        if not isinstance(frame, dict) or "t" not in frame or not isinstance(frame["t"], (int, float)) or isinstance(frame["t"], bool):
            raise ValueError("Each measured frame requires a numeric timestamp t")
        time = float(frame["t"])
        if not math.isfinite(time) or time <= previous or time < 0:
            raise ValueError("Motion timestamps must be finite, nonnegative and strictly increasing")
        previous = time
        visibility = float(frame.get("visibility", 0))
        if not math.isfinite(visibility) or not 0 <= visibility <= 1:
            raise ValueError("Each frame needs a measured visibility from 0 to 1")
        for name in ("shoulder", "elbow", "wrist", "index", "hip"):
            value = frame.get(name)
            if not isinstance(value, list) or len(value) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
                raise ValueError(f"Each frame requires {name}: [x, y, z] in world meters")
    if frames[-1]["t"] - frames[0]["t"] > 180:
        raise ValueError("Trim the motion clip to at most 180 seconds")


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "service": "motionfit", "mujoco_version": mujoco.__version__, "genai_version": getattr(genai, "__version__", "unknown"), "argon_model": ARGON_MODEL}


@app.post("/api/projects")
async def api_create_project(req: CreateProjectRequest) -> Dict[str, Any]:
    if req.design is not None:
        _validate_design(req.design, json.loads((storage.DEMO_DIR / "catalog.json").read_text()))
    project_id = await asyncio.to_thread(create_project, use_demo=req.use_demo, design=req.design)
    return get_project_state(project_id)


@app.get("/api/projects/{project_id}")
async def api_get_project(project_id: str) -> Dict[str, Any]:
    return get_project_state(project_id)


@app.post("/api/projects/{project_id}/task")
async def api_plan_task(project_id: str, req: TaskRequest) -> Dict[str, Any]:
    with _editing(project_id):
        task = await asyncio.to_thread(plan_task, req.prompt, req.constraints, _optional(project_id, "task.json"), req.orientation_constraint)
        design = apply_task_to_design(load_json(project_id, "design.json"), task)
        _validate_design(design, load_json(project_id, "catalog.json"))
        invalidate_results(project_id, "Task requirements changed")
        save_json(project_id, "task.json", task)
        save_json(project_id, "design.json", design)
        landmarks = _optional(project_id, "landmarks.json")
        if landmarks:
            _save_motion(project_id, landmarks, design, task, task["orientation_constraint"])
        return {"status": "interpreted", "task": task, "design": design, "missing_fields": task["missing_fields"], "fingerprint": input_fingerprint(project_id)}


@app.put("/api/projects/{project_id}/design")
async def api_update_design(project_id: str, design: Dict[str, Any]) -> Dict[str, Any]:
    with _editing(project_id):
        _validate_design(design, load_json(project_id, "catalog.json"))
        invalidate_results(project_id, "Design changed")
        save_json(project_id, "design.json", design)
        task = _optional(project_id, "task.json")
        if task:
            task["constraints"] = {key: design.get("payload_kg") if key == "payload_kg" else design["constraints"].get(key) for key in ("payload_kg", "max_cost_usd", "max_total_mass_kg", "min_reach_m")}
            task["missing_fields"] = []
            task["requirements_source"] = "user_edited_design"
            save_json(project_id, "task.json", task)
        landmarks = _optional(project_id, "landmarks.json")
        if landmarks:
            _save_motion(project_id, landmarks, design, task or {}, (task or {}).get("orientation_constraint", "free"))
        return {"status": "updated", "design": design}


@app.post("/api/projects/{project_id}/catalog/files")
async def api_upload_catalog_files(project_id: str, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    with _editing(project_id):
        staged = []
        total = 0
        for upload in files:
            filename = Path(upload.filename or "").name
            extension = Path(filename).suffix.lower()
            if extension not in (".pdf", ".csv") or filename.startswith("."):
                raise ValueError("Upload PDF datasheets or a prices.csv file")
            data = await upload.read(25 * 1024 * 1024 + 1)
            total += len(data)
            if len(data) > 25 * 1024 * 1024 or total > 80 * 1024 * 1024:
                raise ValueError("Catalog uploads exceed the 25 MB file / 80 MB request limit")
            if extension == ".pdf":
                if not data.startswith(b"%PDF"):
                    raise ValueError(f"{filename} is not a PDF")
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(data))
                    if reader.is_encrypted or not 1 <= len(reader.pages) <= 500:
                        raise ValueError("Use an unencrypted PDF with 1 to 500 pages")
                except Exception as exc:
                    raise ValueError(f"{filename}: upload a readable, unencrypted PDF with 1 to 500 pages") from exc
            if extension == ".csv" and filename.lower() != "prices.csv":
                raise ValueError("The price CSV must be named prices.csv")
            staged.append((filename, extension, data))
        if not staged:
            raise ValueError("Choose at least one PDF or prices.csv")
        invalidate_results(project_id, "Catalog source files changed")
        # User PDFs replace the bundled extraction source rather than mixing their origins.
        if any(ext == ".pdf" for _, ext, _ in staged):
            proj_dir = get_project_dir(project_id)
            for old in (proj_dir / "datasheets").glob("*.pdf"):
                old.unlink()
        for filename, extension, data in staged:
            save_bytes(project_id, "prices.csv" if extension == ".csv" else f"datasheets/{filename.lower()}", data, "text/csv" if extension == ".csv" else "application/pdf")
        catalog = load_json(project_id, "catalog.json")
        uploaded_pdfs = [name.lower() for name, extension, _ in staged if extension == ".pdf"]
        if not uploaded_pdfs and not catalog.get("source_files_changed"):
            catalog = normalize_catalog(catalog, prices_path=get_project_dir(project_id) / "prices.csv")
            catalog["price_provenance"] = "user_uploaded_prices.csv"
        catalog["confirmed"] = False
        catalog["source_files_changed"] = bool(uploaded_pdfs) or bool(catalog.get("source_files_changed"))
        if uploaded_pdfs:
            catalog["source_documents"] = uploaded_pdfs
        for part in catalog.get("parts", []):
            part["user_confirmed"] = False
        save_json(project_id, "catalog.json", catalog)
        return {"status": "uploaded", "files": [name for name, _, _ in staged], "requires_extraction": catalog["source_files_changed"]}


@app.post("/api/projects/{project_id}/catalog/extract")
async def api_extract_catalog(project_id: str, force_live: bool = False) -> Dict[str, Any]:
    with _editing(project_id):
        proj_dir = get_project_dir(project_id)
        catalog = _optional(project_id, "catalog.json")
        if catalog and not force_live and not catalog.get("source_files_changed"):
            return {"status": "cached", "catalog": catalog}
        source_documents = (catalog or {}).get("source_documents")
        pdfs = [proj_dir / "datasheets" / name for name in source_documents] if source_documents else sorted((proj_dir / "datasheets").glob("*.pdf"))
        if not pdfs:
            raise ValueError("Upload PDF datasheets before requesting extraction")
        catalog = await asyncio.to_thread(extract_catalog_from_pdfs, pdf_paths=pdfs, prices_path=proj_dir / "prices.csv", cached_catalog_path=None)
        if not catalog or not catalog.get("parts"):
            raise HTTPException(status_code=502, detail="No catalog parts could be extracted; no fallback catalog was accepted")
        catalog["confirmed"] = False
        catalog["source_files_changed"] = False
        catalog["source_documents"] = [pdf.name for pdf in pdfs]
        for part in catalog["parts"]:
            part["user_confirmed"] = False
        invalidate_results(project_id, "Catalog extraction replaced specifications")
        save_json(project_id, "catalog.json", catalog)
        return {"status": "extracted", "catalog": catalog}


@app.put("/api/projects/{project_id}/catalog")
async def api_update_catalog(project_id: str, catalog: Dict[str, Any]) -> Dict[str, Any]:
    with _editing(project_id):
        if catalog.get("source_files_changed") or _optional(project_id, "catalog.json", {}).get("source_files_changed"):
            raise ValueError("Extract the newly uploaded datasheets before confirming the catalog")
        parts = catalog.get("parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(part, dict) for part in parts):
            raise ValueError("Catalog must contain reviewed part objects")
        normalized = normalize_catalog(catalog)
        _validate_catalog(normalized, project_id)
        design = load_json(project_id, "design.json")
        by_id = {part["part_id"]: part for part in normalized["parts"]}
        first_actuator = min((part for part in normalized["parts"] if part["type"] == "actuator"), key=lambda part: (part["mass_kg"]["value"], part["price_usd"]["value"], part["part_id"]))
        baseline_changes = []
        for joint, config in design["joints"].items():
            if by_id.get(config.get("actuator_id"), {}).get("type") != "actuator":
                baseline_changes.append({"joint": joint, "from": config.get("actuator_id"), "to": first_actuator["part_id"]})
                config["actuator_id"] = first_actuator["part_id"]
            if config.get("gearbox_id") and by_id.get(config["gearbox_id"], {}).get("type") != "gearbox":
                config["gearbox_id"] = None
        _validate_design(design, normalized)
        normalized["confirmed"] = True
        for part in normalized["parts"]:
            part["user_confirmed"] = True
        invalidate_results(project_id, "Catalog explicitly reviewed and confirmed")
        save_json(project_id, "catalog.json", normalized)
        if baseline_changes:
            save_json(project_id, "design.json", design)
        workflow = _optional(project_id, "workflow.json", {})
        workflow["catalog_confirmed"] = True
        save_json(project_id, "workflow.json", workflow)
        return {"status": "confirmed", "catalog": normalized, "design": design, "baseline_changes": baseline_changes, "baseline_notice": "Missing initial actuators were assigned the lowest-mass reviewed catalog actuator. This is an unverified starting design, not an optimization result." if baseline_changes else None}


def _save_motion(project_id: str, landmarks: Dict[str, Any], design: Dict[str, Any], task: Dict[str, Any], orientation: str) -> Dict[str, Any]:
    if orientation not in ("free", "keep_level", "keep_upright"):
        raise ValueError("Unknown orientation constraint")
    _validate_landmarks(landmarks)
    summary = summarize_motion(landmarks)
    target = process_landmarks_to_target(landmarks_data=landmarks, design=design, orientation_mode=orientation)
    intent = {"task_summary": task.get("summary", "User-provided recorded arm motion"), "phases": [{"name": "recorded_motion", "t_start": float(target["t"][0]), "t_end": float(target["t"][-1]), "notes": "Measured arm trajectory; object actions were not inferred."}], "orientation_constraint": orientation, "source": "user_task_and_measured_landmarks", "payload_kg": design.get("payload_kg"), "Inferred_payload_note": "Payload mass supplied by the user; not inferred from video."}
    save_json(project_id, "landmarks.json", landmarks)
    save_json(project_id, "target.json", target)
    save_json(project_id, "intent.json", intent)
    return {"status": "processed", "source": landmarks.get("source", "user_landmarks"), "motion_summary": summary, "intent": intent, "target": target}


@app.post("/api/projects/{project_id}/motion")
async def api_process_motion(project_id: str, req: MotionRequest) -> Dict[str, Any]:
    with _editing(project_id):
        design = load_json(project_id, "design.json")
        task = _optional(project_id, "task.json", {})
        landmarks = req.landmarks
        if landmarks is None:
            if not req.preset:
                raise ValueError("Upload measured landmarks; a motion preset is used only when explicitly requested")
            if req.preset not in ("wipe", "pour", "pick_and_place", "water_bottle"):
                raise ValueError("Unknown explicit demonstration preset")
            landmarks = generate_preset_landmarks(preset_name=req.preset, links=design["links"])
            landmarks["source"] = "explicit_demo_preset"
        _validate_landmarks(landmarks)
        # Process fully before invalidating the prior valid result.
        orientation = req.orientation_constraint or task.get("orientation_constraint", "free")
        target = process_landmarks_to_target(landmarks_data=landmarks, design=design, orientation_mode=orientation)
        if not target.get("t"):
            raise ValueError("Motion processing returned an empty target")
        invalidate_results(project_id, "Measured motion changed")
        return _save_motion(project_id, landmarks, design, task, orientation)


def _write_run(record: Dict[str, Any]) -> None:
    save_json(record["project_id"], f"{record['run_id']}.json", record)


def _emit(record: Dict[str, Any], event: Dict[str, Any]) -> None:
    with RUN_GUARD:
        sequence = int(record.get("event_sequence", 0)) + 1
        payload = {**event, "sequence": sequence, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record["event_sequence"] = sequence
        record["events"].append(payload)
        record["events"] = record["events"][-MAX_EVENTS:]
        RUN_EVENTS[record["run_id"]] = record["events"]
        record["progress"] = payload
        _write_run(record)


def _run_job(record: Dict[str, Any], req: RunLoopRequest, inputs: Dict[str, Any], lock: threading.Lock) -> None:
    project_id = record["project_id"]
    try:
        result = run_design_loop(initial_design=inputs["design"], catalog=inputs["catalog"], materials=load_materials(), target=inputs["target"], intent=inputs["intent"], max_iterations=req.max_iterations, optimize_after_pass=req.enable_post_pass_optimization, include_naive_baseline=req.include_naive_baseline, on_event=lambda evt: _emit(record, evt))
        if input_fingerprint(project_id) != record["fingerprint"]:
            raise RuntimeError("Project inputs changed during verification; result discarded")
        result.update(run_id=record["run_id"], input_fingerprint=record["fingerprint"], input_revision=record["revision"])
        result["limitations"] = ["Attached-payload dynamics only; no grasp or contact simulation.", "Mounting compatibility, structural strength, collision avoidance and thermal life require separate engineering checks."]
        final_design = result["final_design"]
        approval = {"approved": False, "engineer_name": None, "notes": None, "approved_at": None}
        report = render_solution_report(project_id=project_id, initial_design=inputs["design"], final_design=final_design, catalog=inputs["catalog"], intent=inputs["intent"], motion_summary=summarize_motion(inputs["landmarks"]), loop_result=result, approval=approval)
        save_json(project_id, "loop_result.json", result)
        save_json(project_id, "design_final.json", final_design)
        save_json(project_id, "iterations.json", result["iterations"])
        save_json(project_id, "approval.json", approval)
        save_text(project_id, "arm_final.xml", build_mjcf(final_design, inputs["catalog"], load_materials()), "application/xml")
        save_text(project_id, "report.html", report, "text/html")
        record.update(status="completed", passed=bool(result["passed"]), completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        RUN_STATUS[record["run_id"]] = "completed"
        _emit(record, {"event": "complete", "type": "complete", "passed": bool(result["passed"]), "final_design_id": final_design["design_id"], "iterations_count": len(result["iterations"])})
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        RUN_STATUS[record["run_id"]] = "failed"
        try:
            _emit(record, {"event": "failed", "type": "failed", "message": record["error"]})
        except Exception:
            pass  # Status remains visibly failed in memory even if remote storage also failed.
    finally:
        lock.release()


@app.post("/api/projects/{project_id}/run")
async def api_run_loop(project_id: str, req: Optional[RunLoopRequest] = None) -> Dict[str, Any]:
    req = req or RunLoopRequest()
    lock = _project_lock(project_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A run or input update is already active for this project")
    try:
        inputs = {key: load_json(project_id, filename) for key, filename in [("design", "design.json"), ("catalog", "catalog.json"), ("target", "target.json"), ("intent", "intent.json"), ("landmarks", "landmarks.json")]}
        task = _optional(project_id, "task.json")
        if project_id != "demo" and (not task or task.get("missing_fields") or task.get("constraints", {}).get("payload_kg") is None):
            raise HTTPException(status_code=422, detail="State the task and payload_kg before verification; video does not establish payload mass")
        catalog = inputs["catalog"]
        if not catalog.get("confirmed") or not all(part.get("user_confirmed") for part in catalog.get("parts", [])):
            raise HTTPException(status_code=409, detail="Review and explicitly confirm the catalog before verification")
        _validate_catalog(catalog, project_id)
        _validate_design(inputs["design"], catalog)
        _validate_landmarks(inputs["landmarks"])
        with RUN_GUARD:
            if sum(record["status"] == "running" for record in RUN_RECORDS.values()) >= 4:
                raise HTTPException(status_code=429, detail="All four verification workers are busy. Retry when a run finishes.")
        workflow = invalidate_results(project_id, "New verification run")
        run_id = f"run-{uuid.uuid4().hex}"
        record = {"run_id": run_id, "project_id": project_id, "status": "running", "fingerprint": input_fingerprint(project_id), "revision": workflow["revision"], "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "events": [], "error": None}
        with RUN_GUARD:
            for old_id in list(RUN_RECORDS):
                if len(RUN_RECORDS) < MAX_CACHED_RUNS:
                    break
                if RUN_RECORDS[old_id]["status"] != "running":
                    RUN_RECORDS.pop(old_id, None)
                    RUN_EVENTS.pop(old_id, None)
                    RUN_STATUS.pop(old_id, None)
                    RUN_FUTURES.pop(old_id, None)
            RUN_RECORDS[run_id] = record
            RUN_STATUS[run_id] = "running"
            RUN_EVENTS[run_id] = record["events"]
        _emit(record, {"event": "queued", "type": "queued", "message": "Starting measured-motion verification"})
        RUN_FUTURES[run_id] = EXECUTOR.submit(_run_job, record, req, inputs, lock)
        return {"run_id": run_id, "project_id": project_id, "status": "running", "fingerprint": record["fingerprint"], "events_url": f"/api/runs/{run_id}/events", "status_url": f"/api/runs/{run_id}"}
    except Exception:
        lock.release()
        raise


def _get_run(run_id: str) -> Dict[str, Any]:
    if not run_id.startswith("run-") or not run_id[4:].isalnum() or len(run_id) > 80:
        raise ValueError("Invalid run ID")
    with RUN_GUARD:
        if run_id in RUN_RECORDS:
            return RUN_RECORDS[run_id]
    for path in storage.PROJECTS_DIR.glob(f"*/{run_id}.json"):
        record = json.loads(path.read_text())
        if record["status"] == "running":
            record.update(status="failed", error="Verification was interrupted by a service restart. Start a new run.")
            _write_run(record)
        return record
    raise HTTPException(status_code=404, detail="Run not found; open its project to restore persisted state")


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: str) -> Dict[str, Any]:
    record = dict(_get_run(run_id))
    if record["status"] == "completed":
        result = _optional(record["project_id"], "loop_result.json")
        record["loop_result"] = result if result and result.get("run_id") == run_id else None
        record["stale"] = record["fingerprint"] != input_fingerprint(record["project_id"])
    return record


@app.get("/api/runs/{run_id}/events")
async def api_stream_run_events(run_id: str, after: int = 0) -> StreamingResponse:
    _get_run(run_id)
    async def generate() -> AsyncGenerator[str, None]:
        cursor = after
        while True:
            record = _get_run(run_id)
            for event in list(record.get("events", [])):
                if event["sequence"] > cursor:
                    cursor = event["sequence"]
                    yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
            if record["status"] in ("completed", "failed"):
                break
            yield ": heartbeat\n\n"
            await asyncio.sleep(.5)
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _current_passing_result(project_id: str) -> Dict[str, Any]:
    result = _optional(project_id, "loop_result.json")
    if not result or not result.get("passed"):
        raise HTTPException(status_code=409, detail="A passing verification result is required")
    if not result.get("run_id") or _get_run(result["run_id"])["status"] != "completed":
        raise HTTPException(status_code=409, detail="The verification run did not complete successfully")
    if not result.get("input_fingerprint") or result["input_fingerprint"] != input_fingerprint(project_id):
        raise HTTPException(status_code=409, detail="Verification is stale; run the current inputs again")
    return result


@app.post("/api/projects/{project_id}/accept")
async def api_accept_design(project_id: str, req: Optional[AcceptRequest] = None) -> Dict[str, Any]:
    with _editing(project_id):
        result = _current_passing_result(project_id)
        if req and req.run_id and req.run_id != result["run_id"]:
            raise HTTPException(status_code=409, detail="The requested candidate is not the current verified run")
        workflow = _optional(project_id, "workflow.json", {})
        workflow.update(accepted=True, accepted_run_id=result["run_id"], accepted_fingerprint=result["input_fingerprint"], accepted_design_id=result["final_design"]["design_id"])
        save_json(project_id, "workflow.json", workflow)
        return {"status": "accepted", "accepted": True, "workflow": workflow, "final_design": result["final_design"], "loop_result": result}


@app.post("/api/projects/{project_id}/approve")
async def api_approve_design(project_id: str, req: ApproveRequest) -> Dict[str, Any]:
    with _editing(project_id):
        result = _current_passing_result(project_id)
        workflow = _optional(project_id, "workflow.json", {})
        if not workflow.get("accepted") or workflow.get("accepted_fingerprint") != result["input_fingerprint"] or workflow.get("accepted_run_id") != result["run_id"]:
            raise HTTPException(status_code=409, detail="Accept the current verified design candidate before approval")
        name = req.engineer_name.strip()
        if not name:
            raise ValueError("Enter the reviewing engineer's name")
        approval = {"approved": True, "engineer_name": name, "notes": (req.notes or "").strip(), "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "input_fingerprint": result["input_fingerprint"], "run_id": result["run_id"], "design_id": result["final_design"]["design_id"]}
        save_json(project_id, "approval.json", approval)
        html = render_solution_report(project_id=project_id, initial_design=load_json(project_id, "design.json"), final_design=result["final_design"], catalog=load_json(project_id, "catalog.json"), intent=load_json(project_id, "intent.json"), motion_summary=summarize_motion(load_json(project_id, "landmarks.json")), loop_result=result, approval=approval)
        save_text(project_id, "report.html", html, "text/html")
        return {"status": "approved", "approval": approval}


@app.get("/api/projects/{project_id}/report", response_class=HTMLResponse)
async def api_get_report(project_id: str) -> HTMLResponse:
    path = get_project_dir(project_id) / "report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run verification to generate a report")
    result = _optional(project_id, "loop_result.json")
    if not result or result.get("input_fingerprint") != input_fingerprint(project_id):
        raise HTTPException(status_code=409, detail="Report is stale; verify current project inputs")
    return HTMLResponse(path.read_text())


@app.get("/api/projects/{project_id}/export")
@app.get("/api/projects/{project_id}/export.zip")
async def api_export_zip(project_id: str) -> Response:
    approval = _optional(project_id, "approval.json", {})
    if not approval.get("approved"):
        raise HTTPException(status_code=403, detail="Engineer approval is required before export")
    result = _current_passing_result(project_id)
    workflow = _optional(project_id, "workflow.json", {})
    if approval.get("input_fingerprint") != result["input_fingerprint"] or approval.get("run_id") != result["run_id"] or not workflow.get("accepted") or workflow.get("accepted_run_id") != result["run_id"]:
        raise HTTPException(status_code=403, detail="Approval is stale; accept and approve the current verified design")
    return Response(build_export_zip(project_id), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="motionfit_{project_id}_verified_mjcf_bundle.zip"'})


@app.get("/api/projects/{project_id}/datasheets/{doc_name}")
async def api_get_datasheet(project_id: str, doc_name: str) -> FileResponse:
    if Path(doc_name).name != doc_name or not doc_name.lower().endswith(".pdf"):
        raise ValueError("Invalid datasheet name")
    path = get_project_dir(project_id) / "datasheets" / doc_name.lower()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Datasheet not found in this project's catalog")
    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
