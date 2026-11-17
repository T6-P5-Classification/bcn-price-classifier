"""Checks for the champion feature-importance artifact."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"
PATH = RESULTS / "champion_feature_importance.csv"


@pytest.mark.skipif(not PATH.exists(), reason="champion importance not generated")
def test_champion_feature_importance_schema_and_provenance():
    raw = PATH.read_text(encoding="utf-8")
    assert raw.startswith("# Feature importance del champion: XGBoost")
    frame = pd.read_csv(PATH, comment="#")
    assert len(frame) == 23
    assert frame["feature"].is_unique
    assert frame["model"].eq("XGBoost champion (trial 34)").all()
    assert frame["n_folds"].eq(5).all()
    assert frame["n_repeats"].eq(10).all()
    assert frame["scoring"].eq("f1_macro").all()
    assert frame["snapshot"].eq("2026-09-17").all()
    assert frame["raw_sha256"].nunique() == 1
    assert frame["code_commit"].nunique() == 1
    assert frame["importance_mean"].between(-1, 1).all()

    metadata_path = RESULTS / "champion_feature_importance.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["champion"] == "XGBoost"
    assert metadata["selected_trial"] == 34
    assert metadata["candidate_csv_params_match"] is True
    assert isinstance(metadata["versions_source"], dict)
    assert isinstance(metadata["versions_installed"], dict)
    assert set(metadata["versions_source"]) == set(metadata["versions_installed"])
    assert metadata["candidate_csv_params_match"] is True
    assert metadata["effective_params_match"] is True


def test_validate_champion_candidate_rejects_csv_param_mismatch():
    import hashlib
    from src.feature_importance import validate_champion_candidate
    from src.preprocessing import load_train_test, RAW_DATA_PATH
    from src.xgboost_model import FIXED_PARAMS, make_pipeline
    X_train, *_ = load_train_test(return_thresholds=True)
    details = json.loads((ROOT / "docs/results/xgboost_details.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(ROOT / "docs/results/xgboost_comparison.csv")
    tampered = comparison.copy()
    params = json.loads(tampered.loc[tampered["model"] == "XGBoost", "params"].iloc[0])
    params["random_state"] = 999
    tampered.loc[tampered["model"] == "XGBoost", "params"] = json.dumps(params, sort_keys=True)
    raw_sha256 = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    thresholds = {"lower": 122.0, "upper": 249.72333333333327}

    with pytest.raises(ValueError, match="candidate CSV differs"):
        validate_champion_candidate(
            details, tampered, X_train, thresholds, raw_sha256,
            fixed_params=FIXED_PARAMS,
        )


def test_validate_effective_params_rejects_mismatch():
    from src.feature_importance import validate_effective_params
    from src.xgboost_model import make_pipeline, FIXED_PARAMS

    params = {"max_depth": 6, "n_estimators": 200}
    pipeline = make_pipeline(params)
    bad_params = dict(params, random_state=999)
    with pytest.raises(ValueError, match="effective pipeline parameters differ"):
        validate_effective_params(pipeline, {**FIXED_PARAMS, **bad_params})