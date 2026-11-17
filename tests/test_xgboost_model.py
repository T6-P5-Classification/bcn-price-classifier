"""Small synthetic checks, not measurements of real-data performance."""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import TrialState, create_trial
from sklearn.exceptions import NotFittedError

from src.preprocessing import TARGET_LABELS
from src.xgboost_model import (
    COMPARISON_COLUMNS,
    FIXED_PARAMS,
    N_TRIALS,
    SEARCH_SPACE,
    comparison_row,
    cv_metrics,
    encode_target,
    make_folds,
    make_pipeline,
    oof_diagnostics,
    run_search,
    select_trial,
    trials_table,
)


@pytest.fixture
def training():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "accommodates": rng.integers(1, 9, size=90).astype(float),
        "room_type": np.tile(["Entire home/apt", "Private room", "Shared room"], 30),
    })
    X.loc[::9, "accommodates"] = np.nan
    y = pd.Series(np.tile([0, 1, 2], 30), index=X.index)
    return X, y, make_folds(X, y)


def test_explicit_label_mapping_not_alphabetical():
    labels = pd.Series(["Premium", "Económico", "Medio"], index=[5, 9, 2])
    encoded = encode_target(labels)
    assert encoded.tolist() == [2, 0, 1]
    assert encoded.index.equals(labels.index)


@pytest.mark.parametrize("values", [["Premium", "Otro", "Medio"], ["Económico", "Medio"], ["Económico", None, "Premium"]])
def test_target_rejects_unknown_missing_or_absent_classes(values):
    with pytest.raises(ValueError, match="three agreed"):
        encode_target(pd.Series(values))


def test_fixed_protocol_and_unfitted_pipeline(training):
    X, _, _ = training
    assert N_TRIALS == 40
    pipe = make_pipeline({"n_estimators": 2})
    assert pipe["model"].get_params()["random_state"] == 42
    assert pipe["model"].get_params()["n_jobs"] == 1
    assert pipe["model"].get_params()["tree_method"] == "hist"
    with pytest.raises(NotFittedError):
        pipe["prep"].transform(X)
    for forbidden in FIXED_PARAMS:
        with pytest.raises(ValueError, match="override"):
            make_pipeline({forbidden: 999})


def test_folds_cover_training_once_and_preserve_alignment(training):
    X, y, folds = training
    seen = np.concatenate([valid for _, valid in folds])
    np.testing.assert_array_equal(np.sort(seen), np.arange(len(X)))
    for fit, valid in folds:
        assert not set(fit) & set(valid)
        assert set(y.iloc[fit]) == set(y.iloc[valid]) == {0, 1, 2}
    for (fit, valid), (other_fit, other_valid) in zip(folds, make_folds(X, y)):
        np.testing.assert_array_equal(fit, other_fit)
        np.testing.assert_array_equal(valid, other_valid)
    with pytest.raises(ValueError, match="identical index"):
        make_folds(X, y.iloc[::-1])


def trial(value, gap, *, state=TrialState.COMPLETE):
    return create_trial(value=value, state=state, user_attrs={"gap": gap})


def test_selection_excludes_best_if_gap_too_high_and_boundary():
    study = optuna.create_study(direction="maximize")
    study.add_trial(trial(0.95, 0.12))
    study.add_trial(trial(0.89, 0.05))  # exactly five points is not eligible
    study.add_trial(trial(0.80, 0.049))
    study.add_trial(trial(0.80, 0.02))  # exact F1 tie chooses earliest
    assert select_trial(study).number == 2


def test_selection_fails_explicitly_without_eligible_trial():
    study = optuna.create_study(direction="maximize")
    study.add_trial(trial(0.9, 0.1))
    study.add_trial(trial(None, -0.01, state=TrialState.FAIL))
    with pytest.raises(ValueError, match="No completed trial"):
        select_trial(study)


def test_negative_gap_is_not_clipped_or_hidden():
    study = optuna.create_study(direction="maximize")
    study.add_trial(trial(0.7, -0.01))
    assert select_trial(study).user_attrs["gap"] == -0.01


def test_cv_and_oof_full_pipeline_on_synthetic_data(training):
    X, y, folds = training
    pipe = make_pipeline({"n_estimators": 2, "max_depth": 2})
    metrics = cv_metrics(pipe, X, y, folds)
    diagnostics = oof_diagnostics(pipe, X, y, folds)
    matrix = np.asarray(diagnostics["confusion_matrix"])
    assert matrix.sum() == len(X)
    np.testing.assert_array_equal(matrix.sum(axis=1), y.value_counts().sort_index())
    assert diagnostics["oof_accuracy"] == pytest.approx(matrix.trace() / matrix.sum())
    assert diagnostics["oof_extreme_error_rate"] == pytest.approx((matrix[0, 2] + matrix[2, 0]) / len(y))
    assert diagnostics["label_order"] == list(TARGET_LABELS)
    assert diagnostics["classification_report"]["macro avg"]["f1-score"] == pytest.approx(diagnostics["oof_f1_macro"])
    assert diagnostics["evaluation_set"] == "train_oof"
    assert len(metrics["folds"]) == 5
    assert metrics["cv_validation_std"] == pytest.approx(np.std([f["f1_validation"] for f in metrics["folds"]], ddof=0))
    assert metrics["gap"] == pytest.approx(metrics["cv_train_f1_macro"] - metrics["cv_validation_f1_macro"])
    # Both CV tools must clone the full pipeline; the supplied one stays unfitted.
    with pytest.raises(NotFittedError):
        pipe["prep"].transform(X)
    row = comparison_row("Synthetic", metrics, diagnostics, params={"n_estimators": 2},
                         search_seconds=1.0, provenance={"snapshot": "synthetic", "code_commit": "synthetic", "raw_sha256": "synthetic"})
    assert set(row) == set(COMPARISON_COLUMNS)
    assert not any(key.startswith("test_") for key in row)
    assert "predictions" not in diagnostics
    assert json.loads(row["params"]) == {"n_estimators": 2}


def test_search_budget_and_records_on_small_synthetic_data(training, monkeypatch):
    X, y, folds = training
    # Shrink only the test search; production keeps its declared 40-trial space.
    monkeypatch.setitem(SEARCH_SPACE, "n_estimators", {"kind": "int", "low": 2, "high": 2})
    monkeypatch.setitem(SEARCH_SPACE, "max_depth", {"kind": "int", "low": 2, "high": 2})
    study, seconds = run_search(X, y, folds, n_trials=2)
    table = trials_table(study)
    assert len(table) == 2
    assert table["state"].tolist() == ["COMPLETE", "COMPLETE"]
    assert seconds > 0
    assert set(table["number"]) == {0, 1}
    assert all(len(t.user_attrs["folds"]) == 5 for t in study.trials)
    with pytest.raises(ValueError, match="positive"):
        run_search(X, y, folds, n_trials=0)


def test_notebook_discards_reserved_objects_without_access(tmp_path):
    """Execute its real loading cell with objects that reject any inspection."""
    import hashlib
    import importlib.metadata
    import platform
    import subprocess
    from src.cleaning import MODEL_EXCLUDED_COLUMNS, MODEL_FEATURES

    class DoNotInspect:
        def __getattribute__(self, name):
            raise AssertionError(f"Reserved test accessed: {name}")

    X = pd.DataFrame(np.zeros((30, 23)), columns=MODEL_FEATURES)
    y = pd.Series(np.tile(TARGET_LABELS, 10))
    data = tmp_path / "synthetic.gz"
    data.write_bytes(b"synthetic; not Airbnb")
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/06_model_xgboost.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == "train-only")
    namespace = {
        "load_train_test": lambda **_: (X, DoNotInspect(), y, DoNotInspect(), {"lower": 1.0, "upper": 2.0}),
        "MODEL_FEATURES": MODEL_FEATURES, "MODEL_EXCLUDED_COLUMNS": MODEL_EXCLUDED_COLUMNS,
        "encode_target": encode_target, "make_folds": make_folds,
        "ROOT": root, "RAW_DATA_PATH": data, "Path": Path,
        "hashlib": hashlib, "subprocess": subprocess, "platform": platform,
        "importlib": importlib, "pd": pd, "TARGET_LABELS": TARGET_LABELS,
        "SEED": 42, "N_TRIALS": 40, "display": lambda _: None,
    }
    exec(compile("".join(cell["source"]), "xgboost/train-only", "exec"), namespace)
    assert "reserved_X" not in namespace and "reserved_y" not in namespace
    assert namespace["X_train"] is X
