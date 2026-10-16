"""Every test runs against temporary projects and never calls a live model."""
import pytest


@pytest.fixture(autouse=True)
def isolated_projects_and_models(monkeypatch, tmp_path):
    from backend import storage
    monkeypatch.setenv("MOTIONFIT_OFFLINE", "1")
    monkeypatch.delenv("MOTIONFIT_GCS_BUCKET", raising=False)
    monkeypatch.setattr(storage, "PROJECTS_DIR", tmp_path / "projects")
    yield
