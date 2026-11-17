"""Storage of app predictions and feedback (#23): the contract other issues use."""

import math

import pandas as pd
import pytest

from src.cleaning import MODEL_FEATURES
from src.feedback_store import (
    FEEDBACK_COLUMNS,
    PREDICTION_COLUMNS,
    load_feedback,
    load_labelled,
    load_predictions,
    price_class,
    save_feedback,
    save_prediction,
)

THRESHOLDS = {"lower": 122.0, "upper": 249.72333333333327}
PROBABILITIES = {"Económico": 0.2, "Medio": 0.65, "Premium": 0.15}


def listing(**changes):
    values = {feature: 1 for feature in MODEL_FEATURES}
    values.update(neighbourhood_cleansed="el Raval", neighbourhood_group_cleansed="Ciutat Vella",
                  property_type="Entire rental unit", room_type="Entire home/apt",
                  latitude=41.38, longitude=2.17, bathrooms_count=1.5)
    values.update(changes)
    return values


def store(tmp_path, predicted="Medio", **changes):
    return save_prediction(listing(**changes), predicted, PROBABILITIES, model_name="XGBoost",
                           model_version="xgboost-trial34-test", thresholds=THRESHOLDS,
                           directory=tmp_path)


# --- Contract -------------------------------------------------------------

def test_columns_are_the_documented_contract():
    assert PREDICTION_COLUMNS[:5] == ["prediction_id", "created_at", "model_name",
                                      "model_version", "variant"]
    assert PREDICTION_COLUMNS[5:28] == list(MODEL_FEATURES)
    assert PREDICTION_COLUMNS[28:] == ["predicted_class", "prob_economico", "prob_medio",
                                       "prob_premium", "threshold_lower", "threshold_upper"]
    assert FEEDBACK_COLUMNS == ["prediction_id", "created_at", "real_price", "real_class", "correct"]
    assert "price" not in PREDICTION_COLUMNS  # the model never receives the price


# --- Real class from the real price ---------------------------------------

@pytest.mark.parametrize("price, expected", [
    (50, "Económico"), (122.0, "Económico"), (122.01, "Medio"),
    (249.72333333333327, "Medio"), (249.73, "Premium"), (900, "Premium"),
])
def test_price_class_uses_the_training_target_rule(price, expected):
    assert price_class(price, THRESHOLDS) == expected


@pytest.mark.parametrize("price", [0, -10, math.nan, math.inf, None, "120"])
def test_invalid_prices_are_rejected(price):
    with pytest.raises(ValueError, match="mayor que 0"):
        price_class(price, THRESHOLDS)


# --- Predictions ------------------------------------------------------------

def test_every_prediction_is_stored_with_inputs_output_and_thresholds(tmp_path):
    first, second = store(tmp_path), store(tmp_path, predicted="Premium", accommodates=6)
    table = load_predictions(tmp_path)
    assert list(table.columns) == PREDICTION_COLUMNS
    assert table["prediction_id"].tolist() == [first, second] and first != second
    assert table["predicted_class"].tolist() == ["Medio", "Premium"]
    assert table["accommodates"].tolist() == [1, 6]
    assert table[["prob_economico", "prob_medio", "prob_premium"]].sum(axis=1).round(6).eq(1).all()
    assert table["variant"].eq("champion").all() and table["model_name"].eq("XGBoost").all()
    assert table["threshold_upper"].iloc[0] == pytest.approx(THRESHOLDS["upper"])


def test_incomplete_predictions_are_rejected(tmp_path):
    values = listing()
    del values["beds"]
    with pytest.raises(ValueError, match="beds"):
        save_prediction(values, "Medio", PROBABILITIES, model_name="m", model_version="v",
                        thresholds=THRESHOLDS, directory=tmp_path)
    with pytest.raises(ValueError, match="class"):
        store(tmp_path, predicted="Lujo")
    with pytest.raises(ValueError, match="three"):
        save_prediction(listing(), "Medio", {"Medio": 1.0}, model_name="m", model_version="v",
                        thresholds=THRESHOLDS, directory=tmp_path)
    assert load_predictions(tmp_path).empty


# --- Feedback ---------------------------------------------------------------

def test_feedback_gets_its_real_class_and_is_joined_to_its_prediction(tmp_path):
    hit, miss, _ = store(tmp_path), store(tmp_path), store(tmp_path)
    assert save_feedback(hit, 180, directory=tmp_path)["correct"] is True
    row = save_feedback(miss, 300, directory=tmp_path)
    assert (row["real_class"], row["correct"]) == ("Premium", False)

    assert len(load_feedback(tmp_path)) == 2
    labelled = load_labelled(tmp_path)
    assert len(labelled) == 2  # the prediction without feedback is not included
    assert set(labelled.columns) == set(PREDICTION_COLUMNS) | {"feedback_at", "real_price",
                                                                "real_class", "correct"}
    assert labelled.set_index("prediction_id").loc[miss, "real_price"] == 300


def test_feedback_is_saved_once_and_only_for_known_predictions(tmp_path):
    prediction_id = store(tmp_path)
    save_feedback(prediction_id, 150, directory=tmp_path)
    with pytest.raises(ValueError, match="Ya se ha guardado"):
        save_feedback(prediction_id, 160, directory=tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        save_feedback("no-existe", 150, directory=tmp_path)
    with pytest.raises(ValueError, match="mayor que 0"):
        save_feedback(store(tmp_path), 0, directory=tmp_path)
    assert len(load_feedback(tmp_path)) == 1


def test_empty_storage_returns_empty_tables_with_the_contract_columns(tmp_path):
    assert list(load_predictions(tmp_path).columns) == PREDICTION_COLUMNS
    assert list(load_feedback(tmp_path).columns) == FEEDBACK_COLUMNS
    assert load_labelled(tmp_path).empty


def test_storage_folder_can_be_set_with_feedback_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path / "custom"))
    store(None)
    assert (tmp_path / "custom" / "predictions.csv").exists()
    assert isinstance(load_predictions(), pd.DataFrame) and len(load_predictions()) == 1
