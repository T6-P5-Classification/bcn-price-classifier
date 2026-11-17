"""Functional contract for the shared classification Pipeline (#31).

The estimator is a small LogisticRegression fixture, not a selected champion.
Synthetic rows test behaviour, never predictive quality on Barcelona data.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.cleaning import MODEL_FEATURES
from src.preprocessing import TARGET_LABELS, build_preprocessor


@pytest.fixture(scope="module")
def fitted_pipeline():
    # 40 rows per class: categorical groups exceed min_frequency=20. One
    # rare property creates the infrequent bucket used by unseen values.
    group = np.repeat(np.arange(3), 40)
    rows = pd.DataFrame({column: group.astype(float) for column in MODEL_FEATURES})
    rows["latitude"] = 41.35 + group * 0.02
    rows["longitude"] = 2.12 + group * 0.02
    for column in (
        "neighbourhood_cleansed", "neighbourhood_group_cleansed",
        "property_type", "room_type",
    ):
        rows[column] = [f"{column}_{value}" for value in group]
    rows.loc[0, "property_type"] = "Rare property"
    for column in [c for c in MODEL_FEATURES if c.startswith(("has_", "is_"))]:
        rows[column] = (group > 0).astype(float)
    target = np.repeat(TARGET_LABELS, 40)
    model = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(rows, target)
    return model, rows


def test_predictions_preserve_row_count_and_use_target_labels(fitted_pipeline):
    model, rows = fitted_pipeline
    sample = rows.iloc[[0, 40, 80]]
    prediction = model.predict(sample)
    assert prediction.shape == (len(sample),)
    assert set(prediction) == set(TARGET_LABELS)
    assert list(model.feature_names_in_) == list(MODEL_FEATURES)
    assert "price" not in model.feature_names_in_


def test_probabilities_are_finite_normalized_and_match_predictions(fitted_pipeline):
    model, rows = fitted_pipeline
    sample = rows.iloc[[0, 40, 80]]
    probabilities = model.predict_proba(sample)
    assert probabilities.shape == (len(sample), len(TARGET_LABELS))
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    # Use the estimator's class order, not the presentation order.
    np.testing.assert_array_equal(
        model.classes_[probabilities.argmax(axis=1)], model.predict(sample)
    )


def test_prediction_handles_missing_values_without_refitting(fitted_pipeline):
    model, rows = fitted_pipeline
    sample = rows.iloc[[0, 40]].copy()
    sample.loc[:, "beds"] = np.nan
    sample.loc[:, "room_type"] = np.nan
    before = sample.copy(deep=True)
    prep = model.named_steps["prep"]
    medians = prep.named_transformers_["num"].named_steps["imputer"].statistics_.copy()
    probabilities = model.predict_proba(sample)
    assert np.isfinite(probabilities).all()
    assert set(model.predict(sample)).issubset(TARGET_LABELS)
    pd.testing.assert_frame_equal(sample, before)
    np.testing.assert_array_equal(
        medians, prep.named_transformers_["num"].named_steps["imputer"].statistics_
    )


def test_unseen_category_and_column_order_are_supported(fitted_pipeline):
    model, rows = fitted_pipeline
    sample = rows.iloc[[40, 80]].copy()
    sample["property_type"] = "Previously unseen property"
    prediction = model.predict(sample)
    assert set(prediction).issubset(TARGET_LABELS)
    np.testing.assert_array_equal(
        prediction, model.predict(sample.loc[:, list(reversed(MODEL_FEATURES))])
    )
    assert np.isfinite(model.predict_proba(sample)).all()


def test_missing_required_feature_is_rejected(fitted_pipeline):
    model, rows = fitted_pipeline
    with pytest.raises(ValueError, match="beds"):
        model.predict(rows.iloc[:1].drop(columns="beds"))


def test_joblib_round_trip_preserves_predictions_and_probabilities(
    fitted_pipeline, tmp_path
):
    model, rows = fitted_pipeline
    # Only load an artifact created by this test, never an untrusted pickle.
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    sample = rows.iloc[[0, 40, 80]]
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    np.testing.assert_array_equal(restored.predict(sample), model.predict(sample))
    np.testing.assert_allclose(restored.predict_proba(sample), model.predict_proba(sample))
