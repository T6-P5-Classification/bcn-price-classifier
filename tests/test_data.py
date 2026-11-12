"""Data-contract checks using synthetic CSVs, without the Airbnb snapshot.

These integration tests complement the cleaning/preprocessing unit tests:
expected prices and row identities are defined independently of production
cleaning, and the real loader reads the temporary compressed CSV.
"""

import numpy as np
import pandas as pd
import pytest

import src.preprocessing as preprocessing
from src.cleaning import MODEL_EXCLUDED_COLUMNS
from src.preprocessing import create_price_target, load_train_test


LABELS = ("Económico", "Medio", "Premium")


@pytest.fixture
def listings_file(tmp_path):
    expected_prices = [float(value) for value in range(30, 120)] + [10000.0]
    raw_prices = [f"${value:,.2f}" for value in expected_prices]
    invalid_prices = [None, "", "not-a-price", "$0.00", "$-5.00", "inf", "-inf"]
    # Interleave invalid rows so retained indices have gaps. Resetting them
    # accidentally would break the association between features and prices.
    for offset, invalid in enumerate(invalid_prices):
        position = 1 + offset * 10
        raw_prices.insert(position, invalid)
        expected_prices.insert(position, np.nan)

    rows = np.arange(len(raw_prices))
    raw = pd.DataFrame(
        {
            "id": rows + 1000,
            "host_id": rows + 2000,
            "name": "Synthetic listing",
            "host_name": "Synthetic host",
            "listing_url": "https://example.invalid/listing",
            "price": raw_prices,
            "price_quote_total_price": 999.0,
            "price_quote_price_per_night": 333.0,
            "estimated_revenue_l365d": 50000.0,
            "latitude": 41.30 + rows / 10000,
            "longitude": 2.10 + rows / 10000,
            "room_type": np.where(rows % 2 == 0, "Entire home/apt", "Private room"),
            "accommodates": rows % 6 + 1,
            "bedrooms": rows % 3 + 1,
            "beds": rows % 4 + 1,
            "minimum_nights": np.where(rows % 5 == 0, np.nan, rows % 8 + 1.0),
            "amenities": np.where(rows % 2 == 0, '["Hair dryer"]', '["Dryer"]'),
            "bathrooms_text": "1 private bath",
            "number_of_reviews": 5,
            "reviews_per_month": 0.5,
            "first_review": "2025-09-01",
            "last_review": "2026-09-01",
            "last_scraped": "2026-09-18",
        }
    )
    path = tmp_path / "listings.csv.gz"
    raw.to_csv(path, index=False, compression="gzip")
    prices = pd.Series(expected_prices, index=raw.index, name="price")
    return path, raw, prices


@pytest.fixture
def prepared_data(listings_file):
    path, _, _ = listings_file
    return load_train_test(path, return_thresholds=True)


def test_only_valid_price_rows_survive_including_positive_outlier(
    listings_file, prepared_data
):
    _, raw, prices = listings_file
    X_train, X_test, _, _, _ = prepared_data
    kept = X_train.index.union(X_test.index)

    pd.testing.assert_index_equal(kept, prices.dropna().index)
    assert prices.loc[kept].gt(0).all()
    assert np.isfinite(prices.loc[kept]).all()
    assert prices.idxmax() in kept
    assert prices.loc[prices.idxmax()] == 10000.0
    assert not set(prices[prices.isna()].index).intersection(kept)
    # Identity is also checked against a raw identifier, which never enters X.
    assert set(raw.loc[X_train.index, "id"]).isdisjoint(raw.loc[X_test.index, "id"])


def test_split_accounts_for_every_valid_row_once(listings_file, prepared_data):
    _, _, prices = listings_file
    X_train, X_test, y_train, y_test, _ = prepared_data
    valid_count = prices.notna().sum()

    assert len(X_test) == int(np.ceil(valid_count * 0.20))
    assert len(X_train) == valid_count - len(X_test)
    assert X_train.index.is_unique and X_test.index.is_unique
    assert X_train.index.intersection(X_test.index).empty
    pd.testing.assert_index_equal(X_train.index, y_train.index)
    pd.testing.assert_index_equal(X_test.index, y_test.index)


def test_row_values_stay_attached_to_original_listings(listings_file, prepared_data):
    _, raw, _ = listings_file
    X_train, X_test, _, _, _ = prepared_data
    for X in (X_train, X_test):
        pd.testing.assert_series_equal(X["accommodates"], raw.loc[X.index, "accommodates"])
        pd.testing.assert_series_equal(X["room_type"], raw.loc[X.index, "room_type"])
        np.testing.assert_allclose(X["latitude"], raw.loc[X.index, "latitude"], atol=1e-12, rtol=0)


def test_feature_schema_matches_and_forbidden_inputs_are_absent(prepared_data):
    X_train, X_test, _, _, _ = prepared_data
    pd.testing.assert_index_equal(X_train.columns, X_test.columns)
    pd.testing.assert_series_equal(X_train.dtypes, X_test.dtypes)
    assert X_train.columns.is_unique
    # Keep explicit sentinels as well as the shared set: removing one from
    # MODEL_EXCLUDED_COLUMNS must not silently weaken this data contract.
    forbidden = MODEL_EXCLUDED_COLUMNS | {
        "price", "id", "host_id", "name", "host_name", "listing_url",
        "price_quote_total_price", "price_quote_price_per_night",
        "estimated_revenue_l365d", "amenities", "bathrooms_text",
        "first_review", "last_review", "last_scraped", "price_segment",
    }
    for X in (X_train, X_test):
        assert forbidden.isdisjoint(X.columns)
        assert {
            "latitude", "longitude", "amenities_count", "has_hair_dryer",
            "has_clothes_dryer", "bathrooms_count", "minimum_nights_was_missing",
        }.issubset(X.columns)


def test_predictor_nulls_are_preserved_for_train_only_imputation(
    listings_file, prepared_data
):
    _, raw, _ = listings_file
    X_train, X_test, _, _, _ = prepared_data
    for X in (X_train, X_test):
        expected = raw.loc[X.index, "minimum_nights"]
        assert expected.isna().any()
        pd.testing.assert_series_equal(X["minimum_nights"], expected)
        np.testing.assert_array_equal(
            X["minimum_nights_was_missing"], expected.isna().astype("int8")
        )


def test_final_thresholds_use_train_prices_not_test_or_full_dataset(
    listings_file, monkeypatch
):
    path, _, prices = listings_file
    observed_inputs = []
    compute_thresholds = preprocessing.compute_target_thresholds

    def record_threshold_input(training_prices):
        observed_inputs.append(training_prices.copy())
        return compute_thresholds(training_prices)

    monkeypatch.setattr(preprocessing, "compute_target_thresholds", record_threshold_input)
    X_train, X_test, _, _, thresholds = load_train_test(path, return_thresholds=True)
    # Different thresholds can still produce identical labels on a small
    # fixture. Check the learning inputs, not only the resulting categories.
    assert len(observed_inputs) == 1
    pd.testing.assert_series_equal(
        observed_inputs[0], prices.loc[X_train.index], check_dtype=False
    )
    expected = np.quantile(prices.loc[X_train.index], [1 / 3, 2 / 3])
    actual = [thresholds["lower"], thresholds["upper"]]

    np.testing.assert_allclose(actual, expected)
    # Ensure this fixture distinguishes the correct implementation from
    # learning the boundaries from test or from the complete dataset.
    assert not np.allclose(expected, np.quantile(prices.loc[X_test.index], [1 / 3, 2 / 3]))
    assert not np.allclose(expected, np.quantile(prices.dropna(), [1 / 3, 2 / 3]))


def test_both_targets_follow_the_same_train_boundaries(listings_file, prepared_data):
    _, _, prices = listings_file
    X_train, _, y_train, y_test, _ = prepared_data
    lower, upper = np.quantile(prices.loc[X_train.index], [1 / 3, 2 / 3])
    for target in (y_train, y_test):
        assert target.notna().all()
        assert target.name == "price_segment"
        assert target.cat.ordered
        assert tuple(target.cat.categories) == LABELS
        assert set(target) == set(LABELS)
        expected = [
            "Económico" if price <= lower else "Medio" if price <= upper else "Premium"
            for price in prices.loc[target.index]
        ]
        assert target.tolist() == expected


def test_provisional_price_groups_are_stratified(listings_file, prepared_data):
    _, _, prices = listings_file
    _, X_test, _, _, _ = prepared_data
    valid_prices = prices.dropna()
    lower, upper = np.quantile(valid_prices, [1 / 3, 2 / 3])
    # These are provisional full-data strata, not the final target labels.
    groups = pd.cut(valid_prices, [-np.inf, lower, upper, np.inf], labels=False)
    total_counts = groups.value_counts().sort_index()
    test_counts = groups.loc[X_test.index].value_counts().sort_index()
    expected_counts = total_counts * len(X_test) / len(valid_prices)

    assert set(test_counts.index) == {0, 1, 2}
    assert (test_counts - expected_counts).abs().le(1).all()


def test_default_split_is_reproducible_and_four_return_contract_is_preserved(
    listings_file, prepared_data
):
    path, _, _ = listings_file
    repeated = load_train_test(path)
    assert len(repeated) == 4
    for first, second in zip(prepared_data[:4], repeated):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)


def test_exact_thresholds_belong_to_the_lower_segment():
    prices = pd.Series(
        [1.0, 100.0, np.nextafter(100.0, np.inf), 200.0, np.nextafter(200.0, np.inf)]
    )
    target = create_price_target(prices, {"lower": 100.0, "upper": 200.0})
    assert target.tolist() == ["Económico", "Económico", "Medio", "Medio", "Premium"]


def test_undividable_prices_fail_instead_of_creating_incomplete_classes(listings_file):
    path, raw, _ = listings_file
    raw["price"] = "$100.00"
    raw.to_csv(path, index=False, compression="gzip")

    with pytest.raises(ValueError, match="three stratification bins"):
        load_train_test(path)
