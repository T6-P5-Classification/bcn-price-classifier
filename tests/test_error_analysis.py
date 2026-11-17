"""Error analysis unit/integration checks with synthetic matrices and rows only."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

import src.error_analysis as analysis
from src.error_analysis import (
    MODELS, CV_DESCRIPTION, analyze_matrix, analyze_predictions, common_folds,
    fixed_pipeline, json_safe, plot_error_comparison, plot_matrices,
    reproduce_candidate, row_percentages, selected_candidates,
)
from src.preprocessing import TARGET_LABELS


@pytest.fixture
def candidate_rows():
    return pd.DataFrame([{
        "model": name, "params": "{}", "source_csv": f"synthetic_{name}.csv",
        "evaluation_set": "train_oof", "cv": CV_DESCRIPTION, "snapshot": "synthetic",
        "raw_sha256": "a" * 64, "gap": 0.02, "cv_validation_f1_macro": 0.7, "oof_accuracy": 0.7,
        "oof_f1_macro": 0.7, "oof_extreme_error_rate": 0.1,
    } for name in MODELS])


@pytest.fixture
def training():
    X = pd.DataFrame({"capacity": np.tile([1.0, 3.0, 6.0], 30),
                      "room_type": np.tile(["A", "B", "C"], 30)}, index=range(10, 100))
    X.loc[10, "capacity"] = np.nan
    y = pd.Series(list(TARGET_LABELS) * 30, index=X.index)
    return X, y


def test_all_candidates_required_dummy_duplicates_are_not_analyzed(candidate_rows):
    dummy = candidate_rows.iloc[[0]].assign(model="DummyClassifier")
    results = pd.concat([candidate_rows.iloc[::-1], dummy, dummy])
    before = results.copy(deep=True)
    candidates = selected_candidates(results)
    assert candidates.model.tolist() == list(MODELS)
    pd.testing.assert_frame_equal(results, before)
    with pytest.raises(ValueError, match="one selected"):
        selected_candidates(candidate_rows.iloc[:-1])
    with pytest.raises(ValueError, match="one selected"):
        selected_candidates(pd.concat([candidate_rows, candidate_rows.iloc[[0]]]))


@pytest.mark.parametrize("field,value", [
    ("evaluation_set", "held_out_test"), ("cv", "different"), ("gap", 0.05),
    ("raw_sha256", "b" * 64), ("snapshot", "other"), ("params", "[]"),
])
def test_incompatible_candidates_are_rejected(candidate_rows, field, value):
    candidate_rows.loc[0, field] = value
    with pytest.raises(ValueError):
        selected_candidates(candidate_rows)


def test_matrix_counts_and_three_different_denominators():
    matrix = np.array([[7, 2, 1], [1, 7, 2], [2, 1, 7]])
    summary, directions = analyze_matrix(matrix)
    assert (summary["n_total"], summary["n_correct"], summary["n_errors"]) == (30, 21, 9)
    assert summary["n_neighbour_errors"] == 6 and summary["n_extreme_errors"] == 3
    assert summary["pct_neighbour_of_all"] == 20 and summary["pct_extreme_of_all"] == 10
    assert summary["pct_neighbour_of_errors"] == pytest.approx(100 * 6 / 9)
    assert len(directions) == 6
    assert directions.n_errors.sum() == 9
    assert directions.groupby("error_type").n_errors.sum().to_dict() == {"extremo": 3, "vecino": 6}
    row = directions[(directions.real_class == "Económico") & (directions.predicted_class == "Medio")].iloc[0]
    assert row.n_errors == 2 and row.n_real_class == 10
    assert row.pct_of_all == pytest.approx(100 * 2 / 30)
    assert row.pct_of_errors == pytest.approx(100 * 2 / 9)
    assert row.pct_of_real_class == 20


def test_perfect_model_has_undefined_error_composition_not_zero():
    summary, directions = analyze_matrix(np.diag([10, 10, 10]))
    assert summary["n_errors"] == 0
    assert np.isnan(summary["pct_neighbour_of_errors"]) and np.isnan(summary["pct_extreme_of_errors"])
    assert directions.pct_of_errors.isna().all()
    assert directions.pct_of_all.eq(0).all()
    assert json_safe(summary)["pct_neighbour_of_errors"] is None
    json.dumps(json_safe(summary), allow_nan=False)


def test_absent_real_class_is_not_a_zero_percent_row():
    matrix = np.diag([3, 0, 3])
    percentages = row_percentages(matrix)
    assert np.isnan(percentages[1]).all()
    _, directions = analyze_matrix(matrix)
    assert directions.loc[directions.real_class == "Medio", "pct_of_real_class"].isna().all()


@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)), np.ones((2, 2)), np.full((3, 3), -1),
    np.full((3, 3), 0.5), np.full((3, 3), np.nan), np.full((3, 3), "text"),
])
def test_invalid_matrices_are_rejected(matrix):
    with pytest.raises(ValueError, match="3x3"):
        analyze_matrix(matrix)


def test_real_and_predicted_direction_is_not_inverted():
    matrix, summary, directions = analyze_predictions(
        ["Económico", "Medio", "Premium"], ["Premium", "Medio", "Medio"]
    )
    assert matrix.tolist() == [[0, 0, 1], [0, 1, 0], [0, 1, 0]]
    assert summary["n_extreme_errors"] == summary["n_neighbour_errors"] == 1
    assert directions[(directions.real_class == "Premium") & (directions.predicted_class == "Económico")].n_errors.iloc[0] == 0
    with pytest.raises(ValueError):
        analyze_predictions(["Económico"], ["Unknown"])
    with pytest.raises(ValueError):
        analyze_predictions(["Económico"], ["Económico", "Medio"])
    with pytest.raises(ValueError, match="index order"):
        analyze_predictions(pd.Series(["Medio", "Premium"], index=[1, 2]), pd.Series(["Medio", "Premium"], index=[2, 1]))


@pytest.mark.parametrize("name", MODELS)
def test_oof_encodings_and_full_pipeline_are_used(training, candidate_rows, monkeypatch, name):
    X, y = training
    row = candidate_rows.set_index("model").loc[name].copy()
    row["model"] = name
    predicted_labels = y.to_numpy().copy()
    predicted_labels[0] = "Premium"
    matrix, summary, _ = analyze_predictions(y, predicted_labels)
    row["oof_accuracy"] = summary["n_correct"] / summary["n_total"]
    row["oof_f1_macro"] = f1_score(y, predicted_labels, average="macro")
    row["oof_extreme_error_rate"] = summary["n_extreme_errors"] / summary["n_total"]
    folds = common_folds(X, y)
    calls = []

    def fake_oof(pipeline, features, target, **kwargs):
        assert list(pipeline.named_steps) == ["prep", "model"]
        assert kwargs["n_jobs"] == 1 and kwargs["method"] == "predict"
        assert features is X
        assert len(kwargs["cv"]) == 5
        calls.append(pipeline)
        if name in {"XGBoost", "RandomForest"}:
            assert target.tolist() == [0, 1, 2] * 30
            return pd.Series(predicted_labels).map({c: i for i, c in enumerate(TARGET_LABELS)}).to_numpy()
        assert target is y
        return predicted_labels

    monkeypatch.setattr(analysis, "cross_val_predict", fake_oof)
    result = reproduce_candidate(row, X, y, folds)
    assert len(calls) == 1
    np.testing.assert_array_equal(result["matrix"], matrix)
    assert "predictions" not in result
    row["oof_accuracy"] = 0.1
    with pytest.raises(ValueError, match="differs from saved"):
        reproduce_candidate(row, X, y, folds)
    # An explicitly authorized discrepancy remains visible, not silently passed.
    separate = reproduce_candidate(row, X, y, folds, allow_metric_mismatch=True)
    assert separate["oof_metrics_match_source"] is False
    assert separate["source_metric_check"]["oof_accuracy"]["source"] == 0.1
    assert separate["source_metric_check"]["oof_accuracy"]["local"] == summary["n_correct"] / len(y)
    assert separate["metric_mismatch_explicitly_allowed"] is True
    with pytest.raises(ValueError, match="exact agreed"):
        reproduce_candidate(row, X, y, folds[::-1])


def test_real_synthetic_oof_leaves_original_pipeline_unfitted(training, monkeypatch):
    X, y = training
    pipeline = fixed_pipeline("LogisticRegression", {"C": 0.5})
    folds = common_folds(X, y)
    predicted = cross_val_predict(pipeline, X, y, cv=folds, n_jobs=1)
    _, summary, _ = analyze_predictions(y, predicted)
    row = pd.Series({"model": "LogisticRegression", "params": '{"C": 0.5}',
                     "oof_accuracy": summary["n_correct"] / len(y),
                     "oof_f1_macro": f1_score(y, predicted, average="macro"),
                     "oof_extreme_error_rate": summary["n_extreme_errors"] / len(y)})
    monkeypatch.setattr(analysis, "fixed_pipeline", lambda *_: pipeline)
    reproduce_candidate(row, X, y, folds)
    with pytest.raises(NotFittedError):
        pipeline["prep"].transform(X)
    with pytest.raises(ValueError, match="aligned"):
        common_folds(X, y.iloc[::-1])


def test_figures_render_even_with_no_errors():
    matrix = np.diag([5, 5, 5])
    summary, _ = analyze_matrix(matrix)
    table = pd.DataFrame([{"model": "Synthetic perfect", **summary}])
    figures = [plot_matrices("Synthetic", matrix),
               plot_error_comparison(table), plot_error_comparison(table, within_errors=True)]
    assert figures[0].axes[1].images[0].get_clim() == (0, 100)
    for fig in figures:
        fig.canvas.draw()
        plt.close(fig)


def test_notebook_loading_discards_protected_test_objects(tmp_path, monkeypatch, candidate_rows):
    import src.preprocessing as preprocessing
    import IPython.display
    from src.cleaning import MODEL_FEATURES

    class Protected:
        def __getattribute__(self, name):
            raise AssertionError(f"Reserved test inspected: {name}")

    data = tmp_path / "synthetic.gz"
    data.write_bytes(b"synthetic, not Airbnb")
    candidate_rows["raw_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    X = pd.DataFrame(np.zeros((30, 23)), columns=MODEL_FEATURES)
    y = pd.Series(list(TARGET_LABELS) * 10)
    monkeypatch.setattr(preprocessing, "load_train_test", lambda **_: (X, Protected(), y, Protected(), {"lower": 1, "upper": 2}))
    monkeypatch.setattr(preprocessing, "RAW_DATA_PATH", data)
    monkeypatch.setattr(IPython.display, "display", lambda _: None)
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/08_comparison.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "error17-load-train")["source"])
    namespace = {"ROOT": root, "results": candidate_rows, "pd": pd, "display": lambda _: None}
    exec(compile(source, "error17/load", "exec"), namespace)
    assert "reserved_X" not in namespace and "reserved_y" not in namespace
    assert namespace["X_train"] is X
    # All new code cells are tagged so the CSV-reader tests remain isolated.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell["id"].startswith("error17-"):
            assert "issue17" in cell["metadata"]["tags"]
            tree = ast.parse("".join(cell["source"]))
            assert not any(isinstance(node, ast.Attribute) and node.attr == "optimize" for node in ast.walk(tree))
