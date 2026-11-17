"""Cleaning rules shared by the data preparation notebooks.

This module deliberately does not remove valid price outliers. Price is the
source of the target, so extreme but positive values can represent genuine
premium listings. Missing values in predictor columns are left for the
training pipeline, except for fields with an unambiguous domain meaning.
"""

from __future__ import annotations

import ast
import re

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
        "neighborhood_overview",
        "host_since",
        "host_response_time",
        "host_response_rate",
        "host_acceptance_rate",
        "host_total_listings_count",
        "host_verifications",
        "bathrooms",
        "instant_bookable",
        "source",
        "description",
        "amenities",
        "bathrooms_text",
        "first_review",
        "last_review",
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

# Whitelist of model inputs (closed list, agreed after the teacher's review:
# 71 columns were too many). The app targets NEW listings, so only information
# a host knows when publishing is allowed: review history, availability /
# occupancy and host history are left out by that product rule, not because
# they carried no signal. Further simplifications (calendar variants of
# nights, host_is_superhost, amenities_count, has_washer, has_clothes_dryer,
# maximum_nights) are small, accepted CV losses. Coordinates stay: their
# effect is not linear (expensive areas). Step-by-step ablation:
# notebooks/02_eda_classes.ipynb (section 6) and docs/decisiones.md.
MODEL_FEATURES = (
    # Location
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    # Listing
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    # Bathrooms
    "bathrooms_count",
    "is_shared_bathroom",
    "is_private_bathroom",
    # Amenities
    "has_air_conditioning",
    "has_heating",
    "has_hair_dryer",
    "has_parking",
    "has_self_checkin",
    "has_dedicated_workspace",
    "has_tv",
    "has_pool",
    "has_elevator",
    "has_dishwasher",
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
    """Return only the agreed model inputs (``MODEL_FEATURES``).

    ``price`` and every other column are left out. Columns that do not exist
    yet are skipped, so the function also works on small test frames.
    """

    columns = [column for column in MODEL_FEATURES if column in listings.columns]
    return listings.loc[:, columns].copy()


def select_map_view(listings: pd.DataFrame) -> pd.DataFrame:
    """Return the location/price view used by the map notebook.

    This is a view in memory, not a second persisted dataset. Coordinates are
    retained for visualization and are also available to model training.
    """

    columns = [column for column in MAP_COLUMNS if column in listings.columns]
    return listings.loc[:, columns].copy()


_KEY_AMENITIES = {
    "air_conditioning": (("air conditioning", "air-conditioning"), ()),
    "heating": (("heating",), ()),
    "washer": (("washer",), ()),
    "hair_dryer": (("hair dryer", "hairdryer"), ()),
    "clothes_dryer": (("dryer",), ("hair",)),
    "parking": (("parking",), ()),
    "self_checkin": (("self check-in", "self check in"), ()),
    "dedicated_workspace": (("dedicated workspace",), ()),
    "tv": (("tv", "television"), ()),
    "pool": (("pool",), ()),
    "elevator": (("elevator", "lift"), ()),
    "dishwasher": (("dishwasher",), ()),
}


def _parse_amenity_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip().strip("\"'") for part in text.strip("[]").split(",") if part.strip()]


def _amenity_present(
    amenities: list[str],
    include_terms: tuple[str, ...],
    exclude_terms: tuple[str, ...] = (),
) -> int:
    for item in amenities:
        normalized = str(item).lower()
        if any(
            re.search(rf"\b{re.escape(term)}\b", normalized) is not None
            for term in include_terms
        ) and not any(
            re.search(rf"\b{re.escape(term)}\b", normalized) is not None
            for term in exclude_terms
        ):
            return 1
    return 0


def _add_amenity_features(listings: pd.DataFrame) -> pd.DataFrame:
    result = listings.copy()
    if "amenities" in result.columns:
        amenity_lists = result["amenities"].map(_parse_amenity_list)
        result["amenities_count"] = amenity_lists.map(len).astype("int64")
        for feature, (include_terms, exclude_terms) in _KEY_AMENITIES.items():
            result[f"has_{feature}"] = amenity_lists.map(
                lambda items, include_terms=include_terms, exclude_terms=exclude_terms: _amenity_present(
                    items, include_terms, exclude_terms
                )
            ).astype("int8")
        result = result.drop(columns=["amenities"])
    elif "amenities_count" not in result.columns:
        result["amenities_count"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    return result


def _parse_bathroom_count(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    if "half" in text:
        return 0.5
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else np.nan


def _add_bathroom_features(listings: pd.DataFrame) -> pd.DataFrame:
    result = listings.copy()
    if "bathrooms_text" in result.columns:
        text = result["bathrooms_text"].astype("string")
        result["bathrooms_count"] = result["bathrooms_text"].map(_parse_bathroom_count)
        result["is_shared_bathroom"] = text.str.contains(
            "shared", case=False, na=False
        ).astype("int8")
        result["is_private_bathroom"] = text.str.contains(
            "private", case=False, na=False
        ).astype("int8")
        result = result.drop(columns=["bathrooms_text"])
    return result


def _add_host_features(listings: pd.DataFrame) -> pd.DataFrame:
    result = listings.copy()
    if {"hosts_time_as_host_years", "hosts_time_as_host_months"}.issubset(result.columns):
        years = result["hosts_time_as_host_years"]
        months = result["hosts_time_as_host_months"]
        tenure = years.fillna(0) + months.fillna(0) / 12
        has_tenure = years.notna() | months.notna()
        result["host_tenure_years"] = tenure.where(has_tenure)
    elif "host_since" in result.columns:
        host_since = pd.to_datetime(result["host_since"], errors="coerce")
        if "last_scraped" in result.columns:
            reference = pd.to_datetime(result["last_scraped"], errors="coerce")
            result["host_tenure_years"] = (reference - host_since).dt.days / 365.25
    return result


def _add_review_features(
    listings: pd.DataFrame, snapshot_date: pd.Timestamp | str | None = None
) -> pd.DataFrame:
    result = listings.copy()
    reference = (
        pd.to_datetime(snapshot_date, errors="coerce")
        if snapshot_date is not None
        else None
    )

    if "last_review" in result.columns:
        last_review = pd.to_datetime(result["last_review"], errors="coerce")
        if reference is None and "last_scraped" in result.columns:
            reference = pd.to_datetime(result["last_scraped"], errors="coerce")
        if reference is None:
            reference = pd.Timestamp("2026-09-18")
        if isinstance(reference, pd.Timestamp):
            days = (reference - last_review).dt.days
        else:
            days = (reference - last_review).dt.days
        if snapshot_date is not None:
            days = days.where(last_review <= reference)
        result["days_since_last_review"] = days
        assert result["days_since_last_review"].dropna().ge(0).all()

    if "first_review" in result.columns:
        first_review = pd.to_datetime(result["first_review"], errors="coerce")
        if reference is None and "last_scraped" in result.columns:
            reference = pd.to_datetime(result["last_scraped"], errors="coerce")
        if reference is None:
            reference = pd.Timestamp("2026-09-18")
        if isinstance(reference, pd.Timestamp):
            days = (reference - first_review).dt.days
        else:
            days = (reference - first_review).dt.days
        if snapshot_date is not None:
            days = days.where(first_review <= reference)
        result["days_since_first_review"] = days

    if {"first_review", "last_review"}.issubset(result.columns):
        first_review = pd.to_datetime(result["first_review"], errors="coerce")
        last_review = pd.to_datetime(result["last_review"], errors="coerce")
        result["review_span_days"] = (last_review - first_review).dt.days

    if "has_reviews" not in result.columns and "number_of_reviews" in result.columns:
        result["has_reviews"] = result["number_of_reviews"].gt(0).astype("int8")

    result = result.drop(columns=["first_review", "last_review"], errors="ignore")
    return result


def engineer_features(
    listings: pd.DataFrame, snapshot_date: pd.Timestamp | str | None = None
) -> pd.DataFrame:
    """Add model-ready features derived from amenities, bathrooms and host data."""

    result = listings.copy()
    result = _add_amenity_features(result)
    result = _add_bathroom_features(result)
    result = _add_host_features(result)
    result = _add_review_features(result, snapshot_date=snapshot_date)
    return result
