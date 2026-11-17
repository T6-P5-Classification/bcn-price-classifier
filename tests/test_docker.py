"""Static checks of the Docker setup (#29); building the image is checked by hand."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(path):
    found = {}
    for line in (ROOT / path).read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def test_app_requirements_use_the_project_versions():
    project, app = pins("requirements.txt"), pins("requirements-app.txt")
    app["xgboost"] = app.pop("xgboost-cpu")  # CPU-only build of the same package
    assert set(app) == {"pandas", "numpy", "scikit-learn", "xgboost", "joblib", "streamlit"}
    for package, version in app.items():
        assert project[package] == version, package


def test_dockerignore_lets_in_only_what_the_app_needs():
    lines = [l.strip() for l in (ROOT / ".dockerignore").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"
    allowed = {l[1:] for l in lines if l.startswith("!")}
    assert allowed == {"requirements-app.txt", ".streamlit/", "src/", "app/", "models/app_model.joblib"}


def test_dockerfile_runs_the_app_safely_on_the_given_port():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.12-slim" in dockerfile
    assert "COPY models/app_model.joblib" in dockerfile
    assert "USER appuser" in dockerfile
    assert "--server.port=${PORT}" in dockerfile and "--server.address=0.0.0.0" in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/_stcore/health" in dockerfile
