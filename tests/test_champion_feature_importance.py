"""Checks for the champion feature-importance artifact."""

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