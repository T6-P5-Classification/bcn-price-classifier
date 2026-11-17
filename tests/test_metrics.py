"""Minimum metrics of the delivered models (#32).

Reads only the saved results in docs/results: no raw data, no retraining and
no new look at the reserved test set. Thresholds and their meaning are in
src/model_quality.py.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.model_quality import (
    MAX_EXTREME_ERROR_RATE,
    MAX_GAP,
    MIN_CLASS_F1,
    MIN_F1_MACRO,
    MIN_MARGIN_OVER_DUMMY,
    quality_problems,
)
from src.preprocessing import TARGET_LABELS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"
GOOD = {"f1_macro": 0.76, "class_f1": {"Económico": 0.88, "Medio": 0.66, "Premium": 0.75},
        "extreme_error_rate": 0.01}


def class_f1(report: dict) -> dict:
    return {label: report[label]["f1-score"] for label in TARGET_LABELS}


def comparison() -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob("*_comparison.csv"))])


# --- Delivered results ----------------------------------------------------------

def test_champion_meets_the_minimums_in_cross_validation():
    details = json.loads((RESULTS / "xgboost_details.json").read_text())
    rows = comparison()
    champion = rows.loc[rows["model"] == "XGBoost"].iloc[0]
    dummy_f1 = rows.loc[rows["model"] == "DummyClassifier", "cv_validation_f1_macro"].max()
    problems = quality_problems(
        champion["cv_validation_f1_macro"],
        class_f1(details["selected_oof"]["classification_report"]),
        champion["oof_extreme_error_rate"],
        gap=champion["gap"],
        dummy_f1_macro=dummy_f1,
    )
    assert problems == []


def test_champion_meets_the_minimums_in_the_single_test_evaluation():
    record = json.loads((RESULTS / "champion_test.json").read_text())
    assert record["champion"] == "XGBoost" and record["evaluated_once"] is True
    problems = quality_problems(
        record["test"]["f1_macro"],
        class_f1(record["classification_report"]),
        record["test"]["extreme_error_rate"],
    )
    assert problems == []


def test_every_compared_model_beats_the_minimum_and_the_dummy():
    rows = comparison()
    dummy_f1 = rows.loc[rows["model"] == "DummyClassifier", "cv_validation_f1_macro"].max()
    for _, row in rows.loc[rows["model"] != "DummyClassifier"].iterrows():
        assert row["cv_validation_f1_macro"] >= MIN_F1_MACRO, row["model"]
        assert row["gap"] < MAX_GAP, row["model"]
        assert row["cv_validation_f1_macro"] - dummy_f1 >= MIN_MARGIN_OVER_DUMMY, row["model"]


def test_app_model_reports_acceptable_test_metrics_if_trained():
    """models/app_model.joblib is generated locally (not in Git)."""
    path = ROOT / "models/app_model.joblib"
    if not path.exists():
        pytest.skip("models/app_model.joblib no existe: python -m src.train_app_model")
    metrics = joblib.load(path).get("test_metrics")
    if metrics is None:
        pytest.skip("el artefacto no incluye métricas de test")
    assert metrics["f1_macro"] >= MIN_F1_MACRO


# --- The check itself -----------------------------------------------------------

def test_a_good_model_has_no_problems():
    assert quality_problems(**GOOD, gap=0.048, dummy_f1_macro=0.167) == []


@pytest.mark.parametrize("change, message", [
    ({"f1_macro": MIN_F1_MACRO - 0.001}, "F1 macro"),
    ({"class_f1": {**GOOD["class_f1"], "Medio": MIN_CLASS_F1 - 0.001}}, "F1 de Medio"),
    ({"extreme_error_rate": MAX_EXTREME_ERROR_RATE + 0.001}, "errores extremos"),
    ({"gap": MAX_GAP}, "gap"),
    ({"dummy_f1_macro": 0.5}, "Dummy"),
])
def test_each_minimum_is_detected(change, message):
    metrics = {**GOOD, "gap": 0.048, "dummy_f1_macro": 0.167, **change}
    problems = quality_problems(**metrics)
    assert len(problems) == 1 and message in problems[0]


def test_boundaries_are_inclusive_for_minimums_and_strict_for_the_gap():
    at_limits = {"f1_macro": MIN_F1_MACRO,
                 "class_f1": {label: MIN_CLASS_F1 for label in TARGET_LABELS},
                 "extreme_error_rate": MAX_EXTREME_ERROR_RATE}
    assert quality_problems(**at_limits) == []
    assert quality_problems(**at_limits, gap=MAX_GAP - 1e-9) == []
    assert quality_problems(**at_limits, gap=MAX_GAP) != []


@pytest.mark.parametrize("change", [
    {"f1_macro": float("nan")},
    {"extreme_error_rate": None},
    {"class_f1": {"Económico": 0.9, "Medio": 0.7}},
])
def test_incomplete_or_invalid_metrics_are_not_accepted(change):
    assert quality_problems(**{**GOOD, **change}) != []


@pytest.mark.parametrize("change", [
    {"f1_macro": 1.5},
    {"class_f1": {**GOOD["class_f1"], "Medio": 2}},
    {"extreme_error_rate": -0.2},
    {"dummy_f1_macro": -1},
    {"gap": 1.5},
    {"f1_macro": True},
    {"class_f1": {**GOOD["class_f1"], "Premium": True}},
    {"extreme_error_rate": False},
])
def test_impossible_metrics_are_not_accepted(change):
    """F1 and rates outside [0, 1], or booleans, must never pass as a good model."""
    metrics = {**GOOD, "gap": 0.048, "dummy_f1_macro": 0.167, **change}
    assert quality_problems(**metrics) != []


def test_exact_limits_of_the_ranges_are_valid():
    perfect = {"f1_macro": 1, "class_f1": {label: 1.0 for label in TARGET_LABELS},
               "extreme_error_rate": 0}
    assert quality_problems(**perfect, gap=0.0, dummy_f1_macro=0.0) == []
