"""New data collected from the app for retraining (#24).

The app stores one row per prediction and, optionally, the real price per
night typed by the host.  This module turns that feedback into a table that
can be used to retrain the model.  It always reads through
``src.feedback_store`` (CSV or MongoDB, depending on configuration) and never
opens the storage files directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cleaning import MODEL_FEATURES
from src.feedback_store import load_labelled
from src.preprocessing import TargetThresholds, compute_target_thresholds, create_price_target

REQUIRED_COLUMNS = list(MODEL_FEATURES) + ["real_price"]


def _coerce_frame(data: Any, directory: Any = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if data is None:
        return load_new_data(directory)
    # A path-like first argument is treated as the storage directory.
    try:
        return load_new_data(data)
    except (TypeError, ValueError, OSError):
        return load_new_data(directory)


def load_new_data(directory: str | Path | None = None) -> pd.DataFrame:
    """Every prediction that already has a real price, ready for retraining.

    Thin wrapper over ``feedback_store.load_labelled``: the 23 model inputs and
    ``real_price`` are preserved, together with the metadata stored by the app.
    """
    return load_labelled(directory)


def validate_new_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Check that a frame has the columns and valid prices needed for retraining."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("new data must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"new data is missing required columns: {missing}")
    if not frame.empty:
        prices = pd.to_numeric(frame["real_price"], errors="coerce")
        if prices.isna().any():
            raise ValueError("real_price must be a finite number greater than 0")
        if not np.isfinite(prices.to_numpy(dtype=float)).all():
            raise ValueError("real_price must be a finite number greater than 0")
        if (prices <= 0).any():
            raise ValueError("real_price must be a finite number greater than 0")
        for column in MODEL_FEATURES:
            if column not in frame.columns:
                raise ValueError(f"new data is missing model input: {column}")
    return frame.copy()


def compute_new_thresholds(prices: pd.Series | list | tuple | np.ndarray) -> TargetThresholds:
    """Thresholds (lower, upper) from a collection of new real prices."""
    series = pd.Series(prices)
    if len(series) < 3:
        raise ValueError("at least three prices are required to compute thresholds")
    return compute_target_thresholds(series)


def prepare_new_data(
    data: Any = None,
    directory: str | Path | None = None,
    thresholds: TargetThresholds | dict | None = None,
    *,
    include_target: bool = True,
    target_column: str = "price_segment",
    keep_real_class: bool = True,
) -> pd.DataFrame:
    """Return a retraining table built from new app data.

    ``thresholds`` define the classes from ``real_price``.  When they are not
    supplied they are computed from the new prices themselves, so callers can
    also combine the result with an existing training set and recompute the
    thresholds afterwards.
    """
    frame = validate_new_data(_coerce_frame(data, directory))
    if frame.empty:
        return frame

    if thresholds is None:
        try:
            thresholds = compute_new_thresholds(frame["real_price"])
        except ValueError:
            if {"threshold_lower", "threshold_upper"}.issubset(frame.columns) and not frame.empty:
                thresholds = {
                    "lower": float(frame["threshold_lower"].iloc[0]),
                    "upper": float(frame["threshold_upper"].iloc[0]),
                }
            else:
                raise
    thresholds = {"lower": float(thresholds["lower"]), "upper": float(thresholds["upper"])}

    result = frame.copy()
    if include_target:
        result[target_column] = create_price_target(frame["real_price"], thresholds)

    columns = list(MODEL_FEATURES) + ["real_price"]
    if keep_real_class and "real_class" in result.columns:
        columns.append("real_class")
    if include_target:
        columns.append(target_column)
    # Preserve any extra requested columns without duplicating them.
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"prepared new data is missing column: {column}")
    return result.loc[:, list(dict.fromkeys(columns))].copy()


def new_training_data(
    data: Any = None,
    directory: str | Path | None = None,
    thresholds: TargetThresholds | dict | None = None,
    *,
    target_column: str = "price_segment",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for retraining from new app data."""
    prepared = prepare_new_data(
        data, directory=directory, thresholds=thresholds,
        include_target=True, target_column=target_column,
    )
    if prepared.empty:
        return pd.DataFrame(columns=MODEL_FEATURES), pd.Series(dtype="object")
    X = prepared.loc[:, list(MODEL_FEATURES)].copy()
    y = prepared[target_column].copy()
    return X, y


# Aliases kept for callers that use other common names.
load_new_training_data = load_new_data
collect_new_data = load_new_data
get_new_data = load_new_data
new_data = load_new_data

prepare_new_training_data = prepare_new_data
build_new_training_data = prepare_new_data
build_new_data = prepare_new_data

compute_thresholds = compute_new_thresholds
compute_thresholds_from_prices = compute_new_thresholds
