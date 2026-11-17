import pandas as pd

from src.cleaning import MODEL_EXCLUDED_COLUMNS, MODEL_FEATURES, select_model_features


def test_model_features_have_no_leakage_or_duplicates():
    assert "price" not in MODEL_FEATURES
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)) == 23
    assert not set(MODEL_FEATURES) & set(MODEL_EXCLUDED_COLUMNS)


def test_select_model_features_keeps_only_allowed_columns_in_order():
    extra = ["price", "id", "number_of_reviews", "availability_365", "new_raw_column"]
    columns = list(reversed(MODEL_FEATURES)) + extra
    listings = pd.DataFrame([[1] * len(columns)], columns=columns)

    selected = select_model_features(listings)

    assert list(selected.columns) == list(MODEL_FEATURES)
    assert not set(extra) & set(selected.columns)
    assert list(listings.columns) == columns  # the input is not modified


def test_select_model_features_skips_missing_columns():
    listings = pd.DataFrame({"room_type": ["Private room"], "price": [100.0]})

    selected = select_model_features(listings)

    assert list(selected.columns) == ["room_type"]
