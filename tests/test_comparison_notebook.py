"""The common notebook discovers compatible per-model CSVs without fitting."""
from pathlib import Path
import json

import pandas as pd
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/08_comparison.ipynb"


def write_result(folder, filename, model="LogisticRegression", **overrides):
    row = dict(
        model=model, cv_train_f1_macro=0.72, cv_validation_f1_macro=0.70,
        cv_validation_std=0.01, gap=0.02, gap_under_005=True, cv_fit_seconds=1.0,
        evaluation_set="train_oof", oof_accuracy=0.90, oof_f1_macro=0.90,
        oof_extreme_error_rate=0.01, oof_seconds=1.0, search_seconds=2.0,
        params='{"C": 1.0}', cv="StratifiedKFold(5, shuffle=True, random_state=42)",
        snapshot="2026-09-17", code_commit="synthetic-commit", raw_sha256="a" * 64,
    )
    row.update(overrides)
    pd.DataFrame([row]).to_csv(folder / filename, index=False)


def run_comparison(root, monkeypatch):
    (root / "src").mkdir(exist_ok=True)
    monkeypatch.chdir(root)
    namespace = {}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "08_comparison.ipynb", "exec"), namespace)
    return namespace["results"]


@pytest.fixture
def result_folder(tmp_path):
    path = tmp_path / "docs/results"
    path.mkdir(parents=True)
    return path


def test_combines_all_model_csvs_and_ranks_by_validation_not_oof(
    tmp_path, result_folder, monkeypatch
):
    write_result(result_folder, "logreg_comparison.csv")
    write_result(result_folder, "lightgbm_comparison.csv", "LightGBM",
                 cv_train_f1_macro=0.82, cv_validation_f1_macro=0.80,
                 oof_accuracy=0.60, oof_f1_macro=0.60)
    (result_folder / "logreg_trials.csv").write_text("not a comparison CSV")
    result = run_comparison(tmp_path, monkeypatch)
    assert result["model"].tolist() == ["LightGBM", "LogisticRegression"]
    assert result["source_csv"].tolist() == ["lightgbm_comparison.csv", "logreg_comparison.csv"]
    assert len(result) == 2


def test_missing_files_produce_clear_error(tmp_path, result_folder, monkeypatch):
    with pytest.raises(FileNotFoundError, match="comparison"):
        run_comparison(tmp_path, monkeypatch)


def test_incompatible_columns_are_rejected(tmp_path, result_folder, monkeypatch):
    write_result(result_folder, "logreg_comparison.csv")
    write_result(result_folder, "lightgbm_comparison.csv", "LightGBM")
    path = result_folder / "lightgbm_comparison.csv"
    frame = pd.read_csv(path).drop(columns="cv_validation_std")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="lightgbm_comparison.csv"):
        run_comparison(tmp_path, monkeypatch)


@pytest.mark.parametrize("column,value", [
    ("evaluation_set", "held_out_test"),
    ("snapshot", "2026-01-01"),
    ("raw_sha256", "b" * 64),
    ("cv", "different folds"),
    ("cv_validation_f1_macro", float("nan")),
    ("gap", 0.50),
    ("gap_under_005", False),
])
def test_incomparable_or_missing_scores_are_rejected(
    tmp_path, result_folder, monkeypatch, column, value
):
    write_result(result_folder, "logreg_comparison.csv")
    write_result(result_folder, "lightgbm_comparison.csv", "LightGBM", **{column: value})
    with pytest.raises(ValueError):
        run_comparison(tmp_path, monkeypatch)


def test_held_out_score_columns_are_rejected(tmp_path, result_folder, monkeypatch):
    write_result(result_folder, "logreg_comparison.csv", test_accuracy=0.95)
    with pytest.raises(ValueError, match="test_"):
        run_comparison(tmp_path, monkeypatch)


def test_same_columns_in_different_order_are_supported(tmp_path, result_folder, monkeypatch):
    write_result(result_folder, "logreg_comparison.csv")
    write_result(result_folder, "lightgbm_comparison.csv", "LightGBM")
    path = result_folder / "lightgbm_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[:, list(reversed(frame.columns))].to_csv(path, index=False)
    result = run_comparison(tmp_path, monkeypatch)
    assert set(result["model"]) == {"LightGBM", "LogisticRegression"}


def test_empty_result_file_is_rejected(tmp_path, result_folder, monkeypatch):
    write_result(result_folder, "logreg_comparison.csv")
    path = result_folder / "logreg_comparison.csv"
    pd.read_csv(path).iloc[:0].to_csv(path, index=False)
    with pytest.raises(ValueError, match="logreg_comparison.csv"):
        run_comparison(tmp_path, monkeypatch)
