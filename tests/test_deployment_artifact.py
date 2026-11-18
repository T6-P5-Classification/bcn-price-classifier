"""Reviewed demo artifact: shipped, never trained at service startup (#33)."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from src.ab_test import challenger_share
from src.cleaning import MODEL_FEATURES
from src.preprocessing import TARGET_LABELS
from src.train_app_model import champion_model, champion_provenance, held_out_summary

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/app_model.joblib"
MANIFEST = ROOT / "models/app_model.manifest.json"


def test_demo_artifact_matches_reviewed_manifest():
    manifest = json.loads(MANIFEST.read_text())
    assert MODEL.name == manifest["file"]
    assert MODEL.stat().st_size == manifest["size_bytes"]
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == manifest["sha256"]


def test_demo_artifact_matches_champion_and_expected_metadata():
    test_demo_artifact_matches_reviewed_manifest()
    artifact = joblib.load(MODEL)
    manifest = json.loads(MANIFEST.read_text())
    assert set(artifact) == {
        "model", "thresholds", "columns", "numeric_columns", "property_types",
        "room_types", "room_types_by_property", "neighbourhoods",
        "hood_to_district", "hood_coords", "model_name", "model_version",
        "provisional", "test_metrics", "provenance", "class_labels",
    }
    assert artifact["model_name"] == manifest["model_name"] == "XGBoost"
    assert artifact["model_version"] == manifest["model_version"]
    assert artifact["provenance"] == champion_provenance()
    assert artifact["thresholds"] == artifact["provenance"]["thresholds"]
    assert artifact["test_metrics"] == held_out_summary(artifact["provenance"])
    assert artifact["columns"] == list(MODEL_FEATURES)
    assert list(artifact["model"].feature_names_in_) == list(MODEL_FEATURES)
    assert list(artifact["class_labels"].values()) == list(TARGET_LABELS)
    expected = champion_model().named_steps["model"].get_params()
    actual = artifact["model"].named_steps["model"].get_params()
    for key, value in expected.items():
        if isinstance(value, float) and pd.isna(value):
            assert pd.isna(actual[key])
        else:
            assert actual[key] == value, key
    assert challenger_share(artifact) == 0
    assert manifest["variants"] == ["champion"]


def test_artifact_dependency_versions_match_docker_requirements():
    manifest = json.loads(MANIFEST.read_text())
    pins = {}
    for line in (ROOT / "requirements-app.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            package, version = line.split("==")
            pins[package.replace("xgboost-cpu", "xgboost")] = version
    for package, version in manifest["dependencies"].items():
        assert pins[package] == version, package


def test_docker_does_not_train_at_build_or_start():
    instructions = "\n".join(
        line for line in (ROOT / "Dockerfile").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "train_app_model" not in instructions
    assert "COPY models/app_model.joblib" in instructions
