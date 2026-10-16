"""Isolated loopback server for browser regression tests; no model/network calls."""
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MOTIONFIT_OFFLINE"] = "1"
os.environ.pop("MOTIONFIT_GCS_BUCKET", None)
from backend import storage
storage.PROJECTS_DIR = Path(tempfile.mkdtemp(prefix="motionfit-browser-"))
import uvicorn
uvicorn.run("backend.main:app", host="127.0.0.1", port=8092, log_level="warning")
