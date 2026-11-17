"""Saved aggregate evidence: direction, denominators, source/local separation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.error_analysis import MODELS, analyze_matrix, row_percentages


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results"


def test_saved_error_tables_match_matrices_and_directions():
    details = json.loads((OUT / "error17_details.json").read_text())
    summary = pd.read_csv(OUT / "error17_summary.csv").set_index("model")
    directions = pd.read_csv(OUT / "error17_directions.csv")
    assert summary.index.tolist() == list(MODELS)
    assert len(directions) == 24
    assert details["evaluation_set"] == "train_oof" and details["nested_cv"] is False
    assert details["test_inspected"] is False
    assert details["hyperparameter_search_executed"] is False
    assert details["full_train_fit_executed"] is False
    assert details["fold_fits_executed"] == 20 and details["model_feature_count"] == 23
    for name in MODELS:
        candidate = details["candidates"][name]
        matrix = np.asarray(candidate["matrix_counts"])
        expected_summary, expected_directions = analyze_matrix(matrix)
        assert matrix.sum() == details["train_rows"]
        for key, value in expected_summary.items():
            assert summary.loc[name, key] == pytest.approx(value)
        actual_directions = directions.loc[directions.model == name, expected_directions.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_directions, expected_directions)
        np.testing.assert_allclose(candidate["matrix_row_percentages"], row_percentages(matrix))
        assert candidate["oof_metrics"]["oof_accuracy"] == pytest.approx(matrix.trace() / matrix.sum())
        assert candidate["oof_metrics"]["oof_extreme_error_rate"] == pytest.approx((matrix[0, 2] + matrix[2, 0]) / matrix.sum())


def test_origin_local_discrepancies_are_visible_and_inputs_unchanged():
    details = json.loads((OUT / "error17_details.json").read_text())
    assert details["authorized_metric_mismatch_models"] == ["RandomForest"]
    assert details["rf_difference_cause_verified"] is False
    summary = pd.read_csv(OUT / "error17_summary.csv").set_index("model")
    for name in MODELS:
        candidate = details["candidates"][name]
        checks = candidate["source_metric_check"]
        source = candidate["source_row"]
        for key, check in checks.items():
            assert check["source"] == source[key]
            assert check["local"] == candidate["oof_metrics"][key]
            assert check["local_minus_source"] == pytest.approx(check["local"] - check["source"])
            assert check["matches"] == bool(np.isclose(check["local"], check["source"], atol=1e-9, rtol=0))
        matches = all(check["matches"] for check in checks.values())
        assert candidate["oof_metrics_match_source"] == matches
        assert summary.loc[name, "oof_metrics_match_source"] == matches
        if not matches:
            assert name == "RandomForest" and candidate["metric_mismatch_explicitly_allowed"]
        # Reconcile the stored original row without updating any source CSV.
        frame = pd.read_csv(OUT / source["source_csv"])
        original = frame.loc[frame.model == name].iloc[0]
        for key in checks:
            assert original[key] == source[key]
    for name, signature in details["provenance"]["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == signature
    for name, signature in details["provenance"]["input_csv_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == signature
