"""Cleaning rules shared by the data preparation notebooks.

This module deliberately does not remove valid price outliers. Price is the
source of the target, so extreme but positive values can represent genuine
premium listings. Missing values in predictor columns are left for the
training pipeline, except for fields with an unambiguous domain meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Identifiers, personal/free-text fields, snapshot metadata, and variables
# derived from price or revenue must not enter model training. Coordinates are
# retained because they can capture fine-grained location effects on price and
# are also needed by the map view. Their contribution should be checked with
# an ablation comparison against a model without coordinates.
MODEL_EXCLUDED_COLUMNS = frozenset(
    {
        "id",
        "listing_url",
        "scrape_id",
        "name",
        "picture_url",
        "host_id",
        "host_url",
        "host_profile_id",
        "host_profile_url",
        "host_name",
        "host_about",
        "host_thumbnail_url",
        "host_picture_url",
        "host_neighbourhood",
        "host_location",
        "license",
        "neighbourhood",
        "price",
        "price_quote_checkin_date",
        "price_quote_checkout_date",
        "price_quote_total_price",
        "price_quote_price_per_night",
        "price_quote_raw",
        "estimated_revenue_l365d",
        "last_scraped",
        "calendar_updated",
        "calendar_last_scraped",
    }
)

MAP_COLUMNS = (
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "price",
    "price_segment",
)


def parse_price(price: pd.Series) -> pd.Series:
    """Convert Airbnb currency strings to numeric euros.

    Invalid, empty, non-finite, zero, and negative values become ``NaN`` so
    that the caller can exclude rows that cannot receive a target label.
    """

    parsed = pd.to_numeric(
        price.astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    parsed = parsed.replace([np.inf, -np.inf], np.nan)
    return parsed.where(parsed > 0)


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Apply the agreed cleaning rules without dropping valid outliers.

    Rows without a valid ``price`` are removed because price is the source of
    the classification target. Missing predictor values are preserved for the
    preprocessing pipeline, with indicators where an imputation has a clear
    semantic meaning.
    """

    cleaned = listings.copy()
    cleaned["price"] = parse_price(cleaned["price"])
    cleaned = cleaned.loc[cleaned["price"].notna()].copy()

    if "beds" in cleaned:
        beds_missing = cleaned["beds"].isna()
        cleaned["beds_was_missing"] = beds_missing.astype("int8")

        # A bedroom normally implies at least one bed. Studios (0 bedrooms)
        # receive one bed; if bedrooms is also unavailable, accommodates is a
        # conservative fallback for this small residual group.
        cleaned.loc[beds_missing & cleaned["bedrooms"].gt(0), "beds"] = cleaned.loc[
            beds_missing & cleaned["bedrooms"].gt(0), "bedrooms"
        ]
        cleaned.loc[beds_missing & cleaned["bedrooms"].eq(0), "beds"] = 1
        fallback = beds_missing & cleaned["beds"].isna() & cleaned["accommodates"].notna()
        cleaned.loc[fallback, "beds"] = np.ceil(
            cleaned.loc[fallback, "accommodates"] / 2
        ).clip(lower=1)

    if "minimum_nights" in cleaned:
        cleaned["minimum_nights_was_missing"] = cleaned["minimum_nights"].isna().astype(
            "int8"
        )
        # Keep NaN here. The train-only median must be learned by the model
        # pipeline after the train/test split to prevent data leakage.

    if "number_of_reviews" in cleaned and "reviews_per_month" in cleaned:
        cleaned["has_reviews"] = cleaned["number_of_reviews"].gt(0).astype("int8")
        no_reviews = cleaned["number_of_reviews"].eq(0) & cleaned[
            "reviews_per_month"
        ].isna()
        cleaned.loc[no_reviews, "reviews_per_month"] = 0.0

    return cleaned


def select_model_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Return model features without identifiers, coordinates, or leakage.

    The target source ``price`` is excluded here, after cleaning has made it
    available for target construction. Missing columns are ignored so this
    function can be used before optional target columns are added.
    """

    return listings.drop(columns=list(MODEL_EXCLUDED_COLUMNS), errors="ignore").copy()


def select_map_view(listings: pd.DataFrame) -> pd.DataFrame:
    """Return the location/price view used by the map notebook.

    This is a view in memory, not a second persisted dataset. Coordinates are
    retained for visualization only and never passed to model training.
    """

    columns = [column for column in MAP_COLUMNS if column in listings.columns]
    return listings.loc[:, columns].copy()
