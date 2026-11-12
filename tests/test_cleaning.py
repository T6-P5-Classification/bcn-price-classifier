import pandas as pd

from src.cleaning import (
    _parse_amenity_list,
    _parse_bathroom_count,
    clean_listings,
    engineer_features,
)


def test_parse_amenity_list_handles_string_and_list():
    assert _parse_amenity_list('["Wifi", "Kitchen"]') == ["Wifi", "Kitchen"]
    assert _parse_amenity_list(["A", "B"]) == ["A", "B"]
    assert _parse_amenity_list(None) == []


def test_half_bath_parsing():
    assert _parse_bathroom_count("Half-bath") == 0.5
    assert _parse_bathroom_count("Shared half-bath") == 0.5
    assert _parse_bathroom_count("2 baths") == 2.0


def test_engineer_features_is_idempotent_and_preserves_rows():
    df = pd.DataFrame(
        {
            "price": ["$100.00", "$200.00"],
            "amenities": ['["Wifi", "Kitchen"]', '["Heating"]'],
            "bathrooms_text": ["1 bath", "2 baths"],
            "hosts_time_as_host_years": [1.0, 2.0],
            "hosts_time_as_host_months": [0.0, 6.0],
            "number_of_reviews": [0, 5],
            "reviews_per_month": [None, 1.0],
            "last_review": ["2026-09-01", "2026-09-10"],
            "first_review": ["2026-08-01", "2026-08-10"],
            "last_scraped": ["2026-09-18", "2026-09-18"],
        }
    )
    cleaned = clean_listings(df)
    first = engineer_features(cleaned)
    second = engineer_features(first)
    assert first.shape == second.shape
    assert "amenities_count" in first.columns
    assert "days_since_last_review" in first.columns
    assert first["days_since_last_review"].dropna().ge(0).all()


def test_future_review_dates_become_nan():
    df = pd.DataFrame(
        {
            "price": ["$100.00"],
            "amenities": ['["Heating"]'],
            "bathrooms_text": ["1 bath"],
            "hosts_time_as_host_years": [1.0],
            "hosts_time_as_host_months": [0.0],
            "number_of_reviews": [1],
            "reviews_per_month": [1.0],
            "last_review": ["2026-09-20"],
            "first_review": ["2026-09-20"],
            "last_scraped": ["2026-09-18"],
        }
    )
    cleaned = clean_listings(df)
    featured = engineer_features(cleaned, snapshot_date="2026-09-18")
    assert featured["days_since_last_review"].isna().all()


def test_amenity_indicators_use_word_boundaries():
    df = pd.DataFrame(
        {
            "price": ["$100.00", "$120.00"],
            "amenities": ['["Heating", "Washer"]', '["TV"]'],
            "bathrooms_text": ["1 bath", "1 bath"],
            "hosts_time_as_host_years": [1.0, 1.0],
            "hosts_time_as_host_months": [0.0, 0.0],
            "number_of_reviews": [1, 1],
            "reviews_per_month": [1.0, 1.0],
            "last_review": ["2026-09-01", "2026-09-01"],
            "first_review": ["2026-08-01", "2026-08-01"],
            "last_scraped": ["2026-09-18", "2026-09-18"],
        }
    )
    cleaned = clean_listings(df)
    featured = engineer_features(cleaned)
    assert featured.loc[0, "has_heating"] == 1
    assert featured.loc[0, "has_washer"] == 1
    assert featured.loc[1, "has_tv"] == 1
    assert featured.loc[1, "has_heating"] == 0
