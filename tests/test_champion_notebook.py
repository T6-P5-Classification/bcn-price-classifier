"""Dry run of notebooks/08b_champion_test.ipynb with synthetic data (#22).

The real held-out test can be evaluated only once, so the whole notebook is
executed here with made-up rows first: if the code had a bug, it would show
up in these tests, not in the single real evaluation.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import src.cleaning  # noqa: F401  (imported before the notebook changes sys.path)
import src.preprocessing as preprocessing
import src.xgboost_model  # noqa: F401
from src.xgboost_model import FIXED_PARAMS
from src.cleaning import MODEL_FEATURES
from src.preprocessing import TARGET_LABELS

matplotlib.use("Agg")
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/08b_champion_test.ipynb"
THRESHOLDS = {"lower": 122.0, "upper": 249.72}
SELECTED = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}


def synthetic_split(n_train=150, n_test=60, seed=0):
    rng = np.random.default_rng(seed)

    def frame(n, start):
        X = pd.DataFrame({
            "neighbourhood_cleansed": rng.choice(["el Raval", "la Sagrada Família"], n),
            "neighbourhood_group_cleansed": rng.choice(["Ciutat Vella", "Eixample"], n),
            "latitude": rng.normal(41.39, 0.01, n),
            "longitude": rng.normal(2.17, 0.01, n),
            "property_type": rng.choice(["Entire rental unit", "Room in hotel"], n),
            "room_type": rng.choice(["Entire home/apt", "Private room"], n),
            "accommodates": rng.integers(1, 7, n),
            "bedrooms": rng.integers(0, 4, n).astype(float),
            "beds": rng.integers(1, 5, n).astype(float),
            "minimum_nights": rng.integers(1, 30, n).astype(float),
            "bathrooms_count": rng.integers(1, 3, n).astype(float),
            "is_shared_bathroom": rng.integers(0, 2, n),
            "is_private_bathroom": rng.integers(0, 2, n),
        }, index=range(start, start + n))
        for column in MODEL_FEATURES:
            if column.startswith("has_"):
                X[column] = rng.integers(0, 2, n)
        y = pd.Series(pd.Categorical(np.resize(TARGET_LABELS, n), categories=TARGET_LABELS),
                      index=X.index)
        return X[list(MODEL_FEATURES)], y

    X_train, y_train = frame(n_train, 0)
    X_test, y_test = frame(n_test, 10_000)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Temporary project root with result files and a fake raw data file."""
    raw = tmp_path / "listings.csv.gz"
    raw.write_bytes(b"synthetic raw data")
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    results = tmp_path / "docs/results"
    results.mkdir(parents=True)
    common = dict(cv_validation_std=0.003, oof_accuracy=0.76, oof_extreme_error_rate=0.01,
                  snapshot="2026-09-17", raw_sha256=sha)
    winner_params = json.dumps({**FIXED_PARAMS, **SELECTED}, sort_keys=True)
    pd.DataFrame([dict(model="XGBoost", cv_validation_f1_macro=0.7631, gap=0.048,
                       params=winner_params, **common)]) \
        .to_csv(results / "xgboost_comparison.csv", index=False)
    pd.DataFrame([dict(model="LightGBM", cv_validation_f1_macro=0.7630, gap=0.049, **common),
                  dict(model="DummyClassifier", cv_validation_f1_macro=0.9, gap=0.0, **common)]) \
        .to_csv(results / "lightgbm_comparison.csv", index=False)
    details = {
        "selected_trial": 34, "test_evaluated": False, "train_rows": 150,
        "thresholds": THRESHOLDS, "provenance": {"raw_sha256": sha},
        "selected_params": dict(SELECTED),
    }
    (results / "xgboost_details.json").write_text(json.dumps(details))

    calls = {"load": 0}

    def fake_load(return_thresholds=False):
        calls["load"] += 1
        return (*synthetic_split(), dict(THRESHOLDS))

    monkeypatch.setattr(preprocessing, "load_train_test", fake_load)
    monkeypatch.setattr(preprocessing, "RAW_DATA_PATH", raw)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "synthetic-commit\n")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")
    return tmp_path, calls


def code_cells():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
            if c["cell_type"] == "code"]


def run_notebook(namespace=None):
    namespace = {} if namespace is None else namespace
    for source in code_cells():
        exec(compile(source, "08b_champion_test.ipynb", "exec"), namespace)
    return namespace


def evaluation_cell():
    (source,) = [s for s in code_cells() if "champion.predict(X_test)" in s]
    return source


def test_full_run_saves_one_complete_record(project):
    root, _ = project
    run_notebook()
    record = json.loads((root / "docs/results/champion_test.json").read_text())
    assert record["champion"] == "XGBoost" and record["evaluated_once"] is True
    assert record["evaluation_set"] == "held_out_test" and record["n_test"] == 60
    assert np.array(record["confusion_matrix"]).sum() == 60
    assert record["label_order"] == list(TARGET_LABELS)
    assert set(record["test"]) == {"accuracy", "f1_macro", "extreme_error_rate"}
    assert all(0 <= value <= 1 for value in record["test"].values())
    assert set(TARGET_LABELS) <= set(record["classification_report"])
    assert record["provenance"]["code_commit"] == "synthetic-commit"


def test_second_run_stops_before_loading_any_data(project):
    root, calls = project
    run_notebook()
    assert calls["load"] == 1
    with pytest.raises(RuntimeError, match="ya existe"):
        run_notebook()
    assert calls["load"] == 1


@pytest.mark.parametrize("change", ["thresholds", "train_rows", "raw_sha256", "champion"])
def test_any_mismatch_stops_before_predicting_test(project, change):
    root, _ = project
    results = root / "docs/results"
    details = json.loads((results / "xgboost_details.json").read_text())
    if change == "thresholds":
        details["thresholds"] = {"lower": 100.0, "upper": 200.0}
    elif change == "train_rows":
        details["train_rows"] = 999
    elif change == "raw_sha256":
        details["provenance"]["raw_sha256"] = "b" * 64
    else:
        frame = pd.read_csv(results / "lightgbm_comparison.csv")
        frame.loc[frame.model == "LightGBM", "cv_validation_f1_macro"] = 0.80
        frame.to_csv(results / "lightgbm_comparison.csv", index=False)
    (results / "xgboost_details.json").write_text(json.dumps(details))
    with pytest.raises(AssertionError):
        run_notebook()
    assert not (results / "champion_test.json").exists()


def test_rerunning_the_evaluation_cell_stops_before_predicting(project):
    """Re-running only the evaluation cell must not predict test again."""
    root, _ = project
    namespace = run_notebook()
    output = root / "docs/results/champion_test.json"
    first = output.read_text()
    calls = {"predict": 0}
    original_predict = namespace["champion"].predict

    def counting_predict(X):
        calls["predict"] += 1
        return original_predict(X)

    namespace["champion"].predict = counting_predict
    with pytest.raises(RuntimeError, match="ya existe"):
        exec(compile(evaluation_cell(), "08b_champion_test.ipynb", "exec"), namespace)
    assert calls["predict"] == 0
    assert output.read_text() == first


def test_parameters_must_match_the_winning_csv_row(project):
    """Review case: CSV winner with max_depth=9 but JSON with max_depth=2."""
    root, _ = project
    results = root / "docs/results"
    frame = pd.read_csv(results / "xgboost_comparison.csv")
    frame.loc[0, "params"] = json.dumps({**FIXED_PARAMS, **SELECTED, "max_depth": 9})
    frame.to_csv(results / "xgboost_comparison.csv", index=False)
    with pytest.raises(AssertionError, match="no coinciden"):
        run_notebook()
    assert not (results / "champion_test.json").exists()


def test_effective_model_parameters_are_checked_before_fitting(project, monkeypatch):
    """If the pipeline builder ignored a parameter, the notebook stops before fitting."""
    root, _ = project
    real_make_pipeline = src.xgboost_model.make_pipeline
    monkeypatch.setattr(src.xgboost_model, "make_pipeline",
                        lambda params: real_make_pipeline({**params, "max_depth": 5}))
    with pytest.raises(AssertionError, match="parámetros del candidato ganador"):
        run_notebook()
    assert not (root / "docs/results/champion_test.json").exists()
