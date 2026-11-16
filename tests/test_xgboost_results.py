"""Consistency checks of aggregate, committed-to-delivery evidence; no raw data."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xgboost_model import COMPARISON_COLUMNS, LABEL_TO_INT, N_TRIALS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"


def test_saved_search_budget_selection_and_source_provenance():
    details = json.loads((RESULTS / "xgboost_details.json").read_text())
    trials = pd.read_csv(RESULTS / "xgboost_trials.csv")
    assert details["n_trials_budget"] == details["n_trials_executed"] == N_TRIALS == 40
    assert details["trial_states"] == {"COMPLETE": 40}
    assert trials["number"].tolist() == list(range(40))
    assert trials["state"].eq("COMPLETE").all()
    assert details["test_evaluated"] is False and details["nested_cv"] is False
    assert details["label_to_int"] == LABEL_TO_INT
    assert len(details["input_schema"]) == 23
    eligible = trials.loc[trials["gap"] < 0.05]
    assert details["n_eligible_trials"] == len(eligible)
    selected = eligible.sort_values(["f1_validation", "number"], ascending=[False, True]).iloc[0]
    assert details["selected_trial"] == selected["number"]
    assert details["selected_params"] == json.loads(selected["params"])
    assert details["selected_cv"]["gap"] < 0.05
    for path, signature in details["provenance"]["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == signature


def test_saved_comparison_and_matrices_are_consistent():
    details = json.loads((RESULTS / "xgboost_details.json").read_text())
    comparison = pd.read_csv(RESULTS / "xgboost_comparison.csv")
    assert comparison.columns.tolist() == COMPARISON_COLUMNS
    assert comparison["model"].tolist() == ["DummyClassifier", "XGBoost"]
    assert comparison["evaluation_set"].eq("train_oof").all()
    assert not any(name.startswith("test_") for name in comparison)
    for name, diagnostics in [("DummyClassifier", details["dummy_oof"]), ("XGBoost", details["selected_oof"])]:
        row = comparison.set_index("model").loc[name]
        matrix = np.asarray(diagnostics["confusion_matrix"])
        assert matrix.shape == (3, 3)
        assert matrix.sum() == details["train_rows"]
        assert row["oof_accuracy"] == pytest.approx(matrix.trace() / matrix.sum())
        assert row["oof_extreme_error_rate"] == pytest.approx((matrix[0, 2] + matrix[2, 0]) / matrix.sum())
        report = diagnostics["classification_report"]
        assert report["macro avg"]["f1-score"] == pytest.approx(row["oof_f1_macro"])
        assert row["gap"] == pytest.approx(row["cv_train_f1_macro"] - row["cv_validation_f1_macro"])
        assert row["gap_under_005"] == (row["gap"] < 0.05)
        for position, label in enumerate(diagnostics["label_order"]):
            assert report[label]["support"] == matrix[position].sum()
            assert report[label]["recall"] == pytest.approx(matrix[position, position] / matrix[position].sum())
