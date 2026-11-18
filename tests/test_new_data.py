"""New data for retraining (#24): collection, validation and preparation."""

import math

import numpy as np
import pandas as pd
import pytest

from src.cleaning import MODEL_FEATURES
from src.feedback_store import save_feedback, save_prediction
from src.new_data import (
    REQUIRED_COLUMNS,
    build_new_data,
    build_new_training_data,
    collect_new_data,
    compute_new_thresholds,
    compute_thresholds,
    get_new_data,
    load_new_data,
    load_new_training_data,
    new_data,
    new_training_data,
    prepare_new_data,
    prepare_new_training_data,
    validate_new_data,
)
from src.preprocessing import TARGET_LABELS, compute_target_thresholds

THRESHOLDS = {"lower": 122.0, "upper": 249.72333333333327}
PROBABILITIES = {"Económico": 0.2, "Medio": 0.65, "Premium": 0.15}


def listing(**changes):
    values = {feature: 1 for feature in MODEL_FEATURES}
    values.update(
        neighbourhood_cleansed="el Raval",
        neighbourhood_group_cleansed="Ciutat Vella",
        property_type="Entire rental unit",
        room_type="Entire home/apt",
        latitude=41.38,
        longitude=2.17,
        bathrooms_count=1.5,
    )
    values.update(changes)
    return values


def store(tmp_path, predicted="Medio", price=180.0, **changes):
    prediction_id = save_prediction(
        listing(**changes),
        predicted,
        PROBABILITIES,
        model_name="XGBoost",
        model_version="xgboost-trial34-test",
        thresholds=THRESHOLDS,
        directory=tmp_path,
    )
    save_feedback(prediction_id, price, directory=tmp_path)
    return prediction_id


def test_load_new_data_returns_labelled_feedback(tmp_path):
    store(tmp_path, price=180.0)
    store(tmp_path, predicted="Premium", price=300.0, accommodates=6)
    frame = load_new_data(tmp_path)
    assert len(frame) == 2
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert "real_price" in frame.columns
    assert frame["real_price"].tolist() == [180.0, 300.0]


def test_load_new_data_empty_storage_has_contract_columns(tmp_path):
    frame = load_new_data(tmp_path)
    assert frame.empty
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)


def test_aliases_load_the_same_data(tmp_path):
    store(tmp_path, price=150.0)
    expected = load_new_data(tmp_path)
    for alias in (load_new_training_data, collect_new_data, get_new_data, new_data):
        pd.testing.assert_frame_equal(alias(tmp_path), expected)


def test_validate_new_data_rejects_missing_columns():
    frame = pd.DataFrame({"real_price": [100.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_new_data(frame)


def test_validate_new_data_rejects_invalid_prices():
    frame = pd.DataFrame({**{c: [1] for c in MODEL_FEATURES}, "real_price": [0.0]})
    with pytest.raises(ValueError, match="greater than 0"):
        validate_new_data(frame)
    frame = pd.DataFrame({**{c: [1] for c in MODEL_FEATURES}, "real_price": [float("nan")]})
    with pytest.raises(ValueError, match="greater than 0"):
        validate_new_data(frame)


def test_validate_new_data_rejects_non_dataframe():
    with pytest.raises(ValueError, match="DataFrame"):
        validate_new_data([1, 2, 3])


def test_compute_new_thresholds_matches_preprocessing():
    prices = [50.0, 120.0, 180.0, 260.0, 320.0, 900.0]
    thresholds = compute_new_thresholds(prices)
    assert thresholds == compute_target_thresholds(pd.Series(prices))
    assert compute_thresholds(prices) == thresholds


def test_compute_new_thresholds_needs_three_prices():
    with pytest.raises(ValueError, match="three prices"):
        compute_new_thresholds([10.0, 20.0])


def test_prepare_new_data_adds_target(tmp_path):
    store(tmp_path, price=180.0)
    store(tmp_path, price=300.0)
    prepared = prepare_new_data(tmp_path)
    assert len(prepared) == 2
    assert set(MODEL_FEATURES).issubset(prepared.columns)
    assert "real_price" in prepared.columns
    assert "price_segment" in prepared.columns
    assert set(prepared["price_segment"]).issubset(set(TARGET_LABELS))


def test_prepare_new_data_empty_returns_empty(tmp_path):
    prepared = prepare_new_data(tmp_path)
    assert prepared.empty


def test_prepare_new_data_uses_supplied_thresholds(tmp_path):
    store(tmp_path, price=180.0)
    thresholds = {"lower": 100.0, "upper": 200.0}
    prepared = prepare_new_data(tmp_path, thresholds=thresholds)
    assert prepared["price_segment"].iloc[0] == "Medio"


def test_prepare_new_data_can_omit_target(tmp_path):
    store(tmp_path, price=180.0)
    prepared = prepare_new_data(tmp_path, include_target=False)
    assert "price_segment" not in prepared.columns
    assert "real_price" in prepared.columns


def test_prepare_aliases(tmp_path):
    store(tmp_path, price=180.0)
    expected = prepare_new_data(tmp_path)
    for alias in (prepare_new_training_data, build_new_training_data, build_new_data):
        pd.testing.assert_frame_equal(alias(tmp_path), expected)


def test_new_training_data_returns_xy(tmp_path):
    store(tmp_path, price=180.0)
    store(tmp_path, price=300.0)
    X, y = new_training_data(tmp_path)
    assert list(X.columns) == list(MODEL_FEATURES)
    assert len(X) == len(y) == 2
    assert set(y).issubset(set(TARGET_LABELS))


def test_new_training_data_empty(tmp_path):
    X, y = new_training_data(tmp_path)
    assert list(X.columns) == list(MODEL_FEATURES)
    assert len(X) == len(y) == 0


def test_prepare_accepts_dataframe(tmp_path):
    store(tmp_path, price=180.0)
    frame = load_new_data(tmp_path)
    prepared = prepare_new_data(frame)
    assert len(prepared) == 1


def test_thresholds_are_recomputed_from_prices(tmp_path):
    store(tmp_path, price=180.0)
    store(tmp_path, price=300.0)
    store(tmp_path, price=900.0)
    frame = load_new_data(tmp_path)
    thresholds = compute_new_thresholds(frame["real_price"])
    prepared = prepare_new_data(frame, thresholds=thresholds)
    assert prepared["price_segment"].notna().all()
