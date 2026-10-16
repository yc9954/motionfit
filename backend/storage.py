"""Project storage manager with local persistence, demo seeding, and optional GCS sync."""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from backend.model_builder import load_materials


DATA_ROOT = Path(__file__).resolve().parent / "data"
DEMO_DIR = DATA_ROOT / "demo"
PROJECTS_DIR = DATA_ROOT / "projects"


DEMO_SEED_FILES = [
    "design.json",
    "catalog.json",
    "prices.csv",
    "landmarks.json",
    "target.json",
    "intent.json",
    "loop_result.json",
    "design_final.json",
    "iterations.json",
    "arm_final.xml",
    "report.html",
    "approval.json",
]


def ensure_demo_project(force_reset: bool = False) -> Path:
    """Ensure the 'demo' project directory exists and is populated from backend/data/demo."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    demo_proj = PROJECTS_DIR / "demo"
    if force_reset or not (demo_proj / "design.json").exists():
        demo_proj.mkdir(parents=True, exist_ok=True)
        for fname in DEMO_SEED_FILES:
            src = DEMO_DIR / fname
            if src.exists():
                shutil.copy2(src, demo_proj / fname)
        ds_src = DEMO_DIR / "datasheets"
        ds_dst = demo_proj / "datasheets"
        if ds_src.exists():
            ds_dst.mkdir(parents=True, exist_ok=True)
            for pdf in ds_src.glob("*.pdf"):
                shutil.copy2(pdf, ds_dst / pdf.name)
    return demo_proj


def create_project(use_demo: bool = True, design: Optional[Dict[str, Any]] = None) -> str:
    """Create a new project directory (or reset 'demo' if use_demo=True and design is None) and return project_id."""
    if use_demo and design is None:
        ensure_demo_project(force_reset=True)
        return "demo"

    ensure_demo_project()
    project_id = f"proj-{uuid.uuid4().hex[:8]}"
    proj_dir = PROJECTS_DIR / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    if use_demo:
        for fname in DEMO_SEED_FILES:
            src = DEMO_DIR / fname
            if src.exists():
                shutil.copy2(src, proj_dir / fname)
        ds_src = DEMO_DIR / "datasheets"
        ds_dst = proj_dir / "datasheets"
        if ds_src.exists():
            ds_dst.mkdir(parents=True, exist_ok=True)
            for pdf in ds_src.glob("*.pdf"):
                shutil.copy2(pdf, ds_dst / pdf.name)

    if design is not None:
        save_json(project_id, "design.json", design)

    return project_id


def get_project_dir(project_id: str) -> Path:
    """Return path to a project directory, auto-initializing 'demo' if requested."""
    if project_id == "demo":
        return ensure_demo_project()
    proj_dir = PROJECTS_DIR / project_id
    if not proj_dir.exists():
        raise FileNotFoundError(f"Project '{project_id}' not found")
    return proj_dir


def load_json(project_id: str, filename: str) -> Dict[str, Any]:
    """Load a JSON file from a project directory."""
    proj_dir = get_project_dir(project_id)
    fpath = proj_dir / filename
    if not fpath.exists():
        raise FileNotFoundError(f"File '{filename}' not found in project '{project_id}'")
    return json.loads(fpath.read_text())


def save_json(project_id: str, filename: str, data: Any) -> Path:
    """Save a JSON file to a project directory (and optionally mirror to GCS)."""
    proj_dir = get_project_dir(project_id)
    fpath = proj_dir / filename
    content = json.dumps(data, indent=2)
    fpath.write_text(content)
    _maybe_sync_to_gcs(project_id, filename, content.encode("utf-8"), "application/json")
    return fpath


def save_text(project_id: str, filename: str, text: str, content_type: str = "text/plain") -> Path:
    """Save a text/HTML/XML file to a project directory."""
    proj_dir = get_project_dir(project_id)
    fpath = proj_dir / filename
    fpath.write_text(text)
    _maybe_sync_to_gcs(project_id, filename, text.encode("utf-8"), content_type)
    return fpath


def get_project_state(project_id: str) -> Dict[str, Any]:
    """Return full project state for the frontend UI."""
    proj_dir = get_project_dir(project_id)
    state: Dict[str, Any] = {
        "project_id": project_id,
        "materials": load_materials(),
    }
    for key, fname in [
        ("design", "design.json"),
        ("catalog", "catalog.json"),
        ("target", "target.json"),
        ("intent", "intent.json"),
        ("loop_result", "loop_result.json"),
        ("approval", "approval.json"),
    ]:
        fpath = proj_dir / fname
        state[key] = json.loads(fpath.read_text()) if fpath.exists() else None

    state["has_report"] = (proj_dir / "report.html").exists()
    return state


def build_export_zip(project_id: str) -> bytes:
    """Build an in-memory ZIP archive containing design_final.json, arm_final.xml, iterations.json, catalog.json, target.json, report.html."""
    proj_dir = get_project_dir(project_id)
    buf = io.BytesIO()
    export_files = [
        "design_final.json",
        "arm_final.xml",
        "iterations.json",
        "catalog.json",
        "target.json",
        "intent.json",
        "report.html",
        "approval.json",
    ]
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in export_files:
            fpath = proj_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
    return buf.getvalue()


def _maybe_sync_to_gcs(project_id: str, filename: str, payload: bytes, content_type: str) -> None:
    """Optional GCS write-through if MOTIONFIT_GCS_BUCKET is set."""
    bucket_name = os.environ.get("MOTIONFIT_GCS_BUCKET", "").strip()
    if not bucket_name:
        return
    try:
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"projects/{project_id}/{filename}")
        blob.upload_from_string(payload, content_type=content_type)
    except Exception:
        # Never block local execution if GCS credentials or bucket are absent
        pass
