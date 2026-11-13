from src.cleaning import MODEL_EXCLUDED_COLUMNS, MODEL_FEATURES


def test_model_features_have_no_leakage_or_duplicates():
    assert "price" not in MODEL_FEATURES
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)) == 23
    assert not set(MODEL_FEATURES) & set(MODEL_EXCLUDED_COLUMNS)
