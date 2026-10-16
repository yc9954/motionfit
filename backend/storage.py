"""Atomic project persistence, isolated inputs, versioning and optional GCS read-through."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from backend.model_builder import load_materials

DATA_ROOT = Path(__file__).resolve().parent / "data"
DEMO_DIR = DATA_ROOT / "demo"
PROJECTS_DIR = DATA_ROOT / "projects"
DEMO_SEED_FILES = ["design.json", "catalog.json", "prices.csv", "landmarks.json", "target.json", "intent.json", "loop_result.json", "design_final.json", "iterations.json", "arm_final.xml", "report.html", "approval.json"]
DERIVED_FILES = ["loop_result.json", "design_final.json", "iterations.json", "arm_final.xml", "report.html", "approval.json"]
INPUT_FILES = ["design.json", "catalog.json", "target.json", "intent.json", "task.json", "landmarks.json"]


class StorageSyncError(RuntimeError):
    """Configured remote persistence failed; never pretend a durable write succeeded."""


def validate_project_id(project_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", project_id):
        raise ValueError("Invalid project ID")
    return project_id


def _relative_filename(filename: str) -> Path:
    rel = Path(filename)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or any(part.startswith(".") for part in rel.parts):
        raise ValueError("Invalid project filename")
    return rel


def _gcs_bucket() -> Any:
    name = os.environ.get("MOTIONFIT_GCS_BUCKET", "").strip()
    if not name:
        return None
    try:
        from google.cloud import storage
        return storage.Client().bucket(name)
    except Exception as exc:
        raise StorageSyncError("Configured GCS storage is unavailable") from exc


def _maybe_sync_to_gcs(project_id: str, filename: str, payload: bytes, content_type: str) -> None:
    bucket = _gcs_bucket()
    if bucket is not None:
        try:
            bucket.blob(f"projects/{project_id}/{filename}").upload_from_string(payload, content_type=content_type)
        except Exception as exc:
            raise StorageSyncError("Project could not be persisted to configured GCS storage") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".motionfit-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _restore_project(project_id: str, proj_dir: Path) -> bool:
    bucket = _gcs_bucket()
    if bucket is None:
        return False
    try:
        prefix = f"projects/{project_id}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            name = blob.name[len(prefix):]
            if name and not name.endswith("/"):
                _atomic_write(proj_dir / _relative_filename(name), blob.download_as_bytes())
        return bool(blobs)
    except Exception as exc:
        raise StorageSyncError("Project could not be restored from configured GCS storage") from exc


def ensure_demo_project(force_reset: bool = False) -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    demo_proj = PROJECTS_DIR / "demo"
    if force_reset or not (demo_proj / "design.json").exists():
        if force_reset and demo_proj.exists():
            shutil.rmtree(demo_proj)
        demo_proj.mkdir(parents=True, exist_ok=True)
        for fname in DEMO_SEED_FILES:
            src = DEMO_DIR / fname
            if src.exists():
                shutil.copy2(src, demo_proj / fname)
        if (DEMO_DIR / "datasheets").exists():
            shutil.copytree(DEMO_DIR / "datasheets", demo_proj / "datasheets", dirs_exist_ok=True)
    return demo_proj


def create_project(use_demo: bool = False, design: Optional[Dict[str, Any]] = None) -> str:
    """Create an isolated workspace. The shared demo is available only via GET /projects/demo."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_id = f"proj-{uuid.uuid4().hex}"
    proj_dir = PROJECTS_DIR / project_id
    proj_dir.mkdir()
    seed_design = copy.deepcopy(design or json.loads((DEMO_DIR / "design.json").read_text()))
    seed_design["design_id"] = f"{project_id}-v0"
    if design is None and not use_demo:
        seed_design["payload_kg"] = 0.0
        seed_design["constraints"].update(max_cost_usd=1e9, max_total_mass_kg=1e9, min_reach_m=0.0)
    save_json(project_id, "design.json", seed_design)
    catalog = json.loads((DEMO_DIR / "catalog.json").read_text())
    catalog["confirmed"] = False
    catalog["spot_check_passed"] = False
    catalog["provenance"] = {"source": "bundled_reference_catalog", "documents": "locally_generated_reference_sheets", "manufacturer_originals": False, "notice": "Bundled values and generated reference PDFs require user review. Derived continuous torque and reference prices are estimates; generated PDFs are not manufacturer originals."}
    for part in catalog.get("parts", []):
        part["user_confirmed"] = False
    save_json(project_id, "catalog.json", catalog)
    if (DEMO_DIR / "datasheets").exists():
        for src in (DEMO_DIR / "datasheets").iterdir():
            if src.is_file() and src.suffix.lower() in (".pdf", ".csv"):
                save_bytes(project_id, f"datasheets/{src.name}", src.read_bytes(), "application/pdf" if src.suffix.lower() == ".pdf" else "text/csv")
    prices = DEMO_DIR / "prices.csv"
    if not prices.exists():
        prices = DEMO_DIR / "datasheets" / "prices.csv"
    if prices.exists():
        save_bytes(project_id, "prices.csv", prices.read_bytes(), "text/csv")
    save_json(project_id, "workflow.json", {"revision": 1, "accepted": False, "source": "explicit_demo_inputs" if use_demo else "user_inputs", "catalog_confirmed": False})
    # Explicit demo mode copies input motion only, never approvals or results.
    if use_demo:
        for fname in ("landmarks.json", "target.json", "intent.json"):
            if (DEMO_DIR / fname).exists():
                save_bytes(project_id, fname, (DEMO_DIR / fname).read_bytes(), "application/json")
        save_json(project_id, "task.json", {"prompt": "Explicit built-in demonstration", "summary": "Explicit built-in demonstration", "constraints": {"payload_kg": seed_design["payload_kg"]}, "missing_fields": [], "orientation_constraint": "free", "interpretation_source": "explicit_demo"})
    return project_id


def get_project_dir(project_id: str) -> Path:
    validate_project_id(project_id)
    if project_id == "demo":
        return ensure_demo_project()
    proj_dir = PROJECTS_DIR / project_id
    if not proj_dir.exists() and not _restore_project(project_id, proj_dir):
        raise FileNotFoundError(f"Project '{project_id}' not found")
    return proj_dir


def load_json(project_id: str, filename: str) -> Dict[str, Any]:
    fpath = get_project_dir(project_id) / _relative_filename(filename)
    if not fpath.exists():
        bucket = _gcs_bucket()
        if bucket is not None:
            try:
                blob = bucket.blob(f"projects/{project_id}/{filename}")
                if blob.exists():
                    _atomic_write(fpath, blob.download_as_bytes())
            except Exception as exc:
                raise StorageSyncError("Project file could not be restored from GCS") from exc
    if not fpath.exists():
        raise FileNotFoundError(f"File '{filename}' not found in project '{project_id}'")
    return json.loads(fpath.read_text())


def save_bytes(project_id: str, filename: str, payload: bytes, content_type: str = "application/octet-stream") -> Path:
    fpath = get_project_dir(project_id) / _relative_filename(filename)
    _maybe_sync_to_gcs(project_id, filename, payload, content_type)
    _atomic_write(fpath, payload)
    return fpath


def _sanitize_for_json(obj: Any) -> Any:
    import math
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_json(project_id: str, filename: str, data: Any) -> Path:
    sanitized = _sanitize_for_json(data)
    return save_bytes(project_id, filename, json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False).encode(), "application/json")


def save_text(project_id: str, filename: str, text: str, content_type: str = "text/plain") -> Path:
    return save_bytes(project_id, filename, text.encode(), content_type)


def input_fingerprint(project_id: str) -> str:
    proj_dir = get_project_dir(project_id)
    inputs = {name: json.loads((proj_dir / name).read_text()) if (proj_dir / name).exists() else None for name in INPUT_FILES}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def invalidate_results(project_id: str, reason: str) -> Dict[str, Any]:
    """Invalidate approval before changing inputs; also remove remote derived files."""
    proj_dir = get_project_dir(project_id)
    state_path = proj_dir / "workflow.json"
    workflow = json.loads(state_path.read_text()) if state_path.exists() else {}
    workflow.update(revision=int(workflow.get("revision", 0)) + 1, accepted=False, accepted_run_id=None, accepted_fingerprint=None, invalidation_reason=reason)
    save_json(project_id, "workflow.json", workflow)
    # A false approval marker makes failures in later cleanup fail closed.
    save_json(project_id, "approval.json", {"approved": False, "reason": reason})
    bucket = _gcs_bucket()
    for filename in DERIVED_FILES:
        if filename == "approval.json":
            continue
        if bucket is not None:
            try:
                blob = bucket.blob(f"projects/{project_id}/{filename}")
                if blob.exists():
                    blob.delete()
            except Exception as exc:
                raise StorageSyncError("Stale remote result could not be invalidated") from exc
        (proj_dir / filename).unlink(missing_ok=True)
    return workflow


def get_project_state(project_id: str) -> Dict[str, Any]:
    proj_dir = get_project_dir(project_id)
    state: Dict[str, Any] = {"project_id": project_id, "materials": load_materials()}
    for key, fname in [("design", "design.json"), ("catalog", "catalog.json"), ("landmarks", "landmarks.json"), ("target", "target.json"), ("intent", "intent.json"), ("task", "task.json"), ("loop_result", "loop_result.json"), ("final_design", "design_final.json"), ("approval", "approval.json"), ("workflow", "workflow.json")]:
        fpath = proj_dir / fname
        state[key] = json.loads(fpath.read_text()) if fpath.exists() else None
    state["fingerprint"] = input_fingerprint(project_id)
    state["has_report"] = (proj_dir / "report.html").exists()
    state["accepted"] = bool((state.get("workflow") or {}).get("accepted"))
    return state


def build_export_zip(project_id: str) -> bytes:
    proj_dir = get_project_dir(project_id)
    buf = io.BytesIO()
    files = ["design_final.json", "arm_final.xml", "iterations.json", "catalog.json", "target.json", "intent.json", "task.json", "report.html", "approval.json", "workflow.json"]
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in files:
            if (proj_dir / fname).exists():
                zf.write(proj_dir / fname, arcname=fname)
    return buf.getvalue()
