"""Synthetic checks for train-only feature importance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.feature_importance import (
    IMPORTANCE_COLUMNS,
    N_REPEATS,
    permutation_feature_importance,
)


@pytest.fixture
def training():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "a": rng.normal(size=60),
        "b": rng.normal(size=60),
        "c": rng.integers(0, 4, size=60).astype(float),
    })
    y = pd.Series(np.tile([0, 1, 2], 20), index=X.index)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    return X, y, folds


PROVENANCE = {
    "snapshot": "synthetic",
    "code_commit": "synthetic",
    "raw_sha256": "a" * 64,
}


def test_feature_importance_schema_and_sorting(training):
    X, y, folds = training
    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    frame = permutation_feature_importance(
        pipeline, X, y, folds, model_name="Synthetic",
        n_repeats=3, provenance=PROVENANCE,
    )
    assert list(frame.columns) == IMPORTANCE_COLUMNS
    assert frame["model"].eq("Synthetic").all()
    assert set(frame["feature"]) == set(X.columns)
    assert frame["n_folds"].eq(5).all()
    assert frame["n_repeats"].eq(3).all()
    assert frame["scoring"].eq("f1_macro").all()
    assert frame["importance_mean"].notna().all()
    assert frame["importance_mean"].tolist() == sorted(
        frame["importance_mean"].tolist(), reverse=True
    )


def test_feature_importance_is_reproducible(training):
    X, y, folds = training
    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    first = permutation_feature_importance(
        pipeline, X, y, folds, model_name="Synthetic",
        n_repeats=3, provenance=PROVENANCE,
    )
    second = permutation_feature_importance(
        pipeline, X, y, folds, model_name="Synthetic",
        n_repeats=3, provenance=PROVENANCE,
    )
    pd.testing.assert_frame_equal(first, second)


def test_feature_importance_validates_inputs(training):
    X, y, folds = training
    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_feature_importance(
            pipeline, X, y, folds, model_name="Synthetic",
            n_repeats=0, provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="provenance"):
        permutation_feature_importance(
            pipeline, X, y, folds, model_name="Synthetic",
            n_repeats=1, provenance=None,
        )
    with pytest.raises(ValueError, match="provenance is missing"):
        permutation_feature_importance(
            pipeline, X, y, folds, model_name="Synthetic",
            n_repeats=1,         provenance={"snapshot": "synthetic"},
        )