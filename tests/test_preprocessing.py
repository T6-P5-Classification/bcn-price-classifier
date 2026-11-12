from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import (
    RAW_DATA_PATH,
    TARGET_LABELS,
    compute_target_thresholds,
    create_price_target,
    load_train_test,
)


def test_target_thresholds_are_computed_from_train_values_only():
    train = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    thresholds = compute_target_thresholds(train)
    target = create_price_target(train, thresholds)

    assert thresholds["lower"] == pytest.approx(26.666666666666668)
    assert thresholds["upper"] == pytest.approx(43.33333333333333)
    assert list(target.cat.categories) == list(TARGET_LABELS)
    assert target.notna().all()


def test_load_train_test_is_stratified_and_excludes_price(tmp_path):
    prices = [float(value) for value in range(10, 70)]
    raw = pd.DataFrame(
        {
            "price": [f"${value:.2f}" for value in prices],
            "accommodates": [2] * len(prices),
            "bedrooms": [1] * len(prices),
            "beds": [1] * len(prices),
        }
    )
    path = tmp_path / "listings.csv.gz"
    raw.to_csv(path, index=False, compression="gzip")

    X_train, X_test, y_train, y_test, thresholds = load_train_test(
        path, test_size=0.2, random_state=42, return_thresholds=True
    )

    assert len(X_train) == 48
    assert len(X_test) == 12
    assert not {"price"}.intersection(X_train.columns)
    assert not set(X_train.index).intersection(X_test.index)
    assert set(y_train.unique()) == set(TARGET_LABELS)
    assert set(y_test.unique()) == set(TARGET_LABELS)
    assert thresholds["lower"] < thresholds["upper"]


def test_default_data_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    expected_path = (
        Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.csv.gz"
    )
    raw = pd.DataFrame({"price": [f"${value:.2f}" for value in range(10, 70)]})

    def read_csv(path, **kwargs):
        assert path == expected_path
        assert path.is_absolute()
        return raw.copy()

    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    monkeypatch.chdir(notebooks)
    monkeypatch.setattr(pd, "read_csv", read_csv)

    assert RAW_DATA_PATH == expected_path
    X_train, X_test, y_train, y_test = load_train_test()
    assert len(X_train) == len(y_train) == 48
    assert len(X_test) == len(y_test) == 12
