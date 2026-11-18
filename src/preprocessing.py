"""Shared data preparation for the model notebooks.

- load_train_test() (#12): load, clean, split and create the target.
- build_preprocessor() (#13): unfitted ColumnTransformer, always used
  inside a Pipeline so it learns only from train (or each CV training fold).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import clean_listings, engineer_features, select_model_features


RAW_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.csv.gz"
)
TARGET_LABELS = ("Económico", "Medio", "Premium")


class TargetThresholds(TypedDict):
    """Thresholds needed to reproduce the target on future data."""

    lower: float
    upper: float


def compute_target_thresholds(price_train: pd.Series) -> TargetThresholds:
    """Compute the two class boundaries using training prices only."""

    q1, q2 = price_train.quantile([1 / 3, 2 / 3]).tolist()
    if not np.isfinite([q1, q2]).all() or q1 >= q2:
        raise ValueError("price_train must contain at least three ordered price values")
    return {"lower": float(q1), "upper": float(q2)}


def create_price_target(
    price: pd.Series, thresholds: TargetThresholds
) -> pd.Series:
    """Map prices to ordered Economic/Medium/Premium classes."""

    bins = [-np.inf, thresholds["lower"], thresholds["upper"], np.inf]
    target = pd.cut(
        price,
        bins=bins,
        labels=TARGET_LABELS,
        include_lowest=True,
        ordered=True,
    )
    if target.isna().any():
        raise ValueError("Every price must map to one target class")
    return target.rename("price_segment")


def load_train_test(
    data_path: str | Path = RAW_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    return_thresholds: bool = False,
):
    """Load, prepare, split, and label the listings dataset.

    The provisional bins used by ``train_test_split`` only preserve a similar
    price distribution in both partitions. They are not the final target:
    final thresholds are learned from ``price_train`` only and reused for
    ``y_test``. By default the function returns the four objects required by
    the model notebooks. Set ``return_thresholds=True`` when the boundaries
    must be persisted with a trained model for future predictions.
    """

    raw = pd.read_csv(data_path, compression="gzip")
    cleaned = clean_listings(raw)
    featured = engineer_features(cleaned)
    price = featured["price"].copy()

    # This is only a stratification aid. The final target boundaries below
    # are calculated strictly from the training partition.
    provisional_bins = pd.qcut(price, q=3, labels=False, duplicates="drop")
    if provisional_bins.nunique(dropna=True) != 3:
        raise ValueError("Price cannot be divided into three stratification bins")

    train_index, test_index = train_test_split(
        featured.index,
        test_size=test_size,
        random_state=random_state,
        stratify=provisional_bins.to_numpy(),
    )
    train_index = pd.Index(train_index)
    test_index = pd.Index(test_index)

    thresholds = compute_target_thresholds(price.loc[train_index])
    y_train = create_price_target(price.loc[train_index], thresholds)
    y_test = create_price_target(price.loc[test_index], thresholds)

    model_data = select_model_features(featured)
    X_train = model_data.loc[train_index].copy()
    X_test = model_data.loc[test_index].copy()

    if set(X_train.index) & set(X_test.index):
        raise AssertionError("Train and test indices overlap")
    if "price" in X_train.columns or "price" in X_test.columns:
        raise AssertionError("price must not be a model feature")

    result = (X_train, X_test, y_train, y_test)
    if return_thresholds:
        return (*result, thresholds)
    return result


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median imputation + scaling. Categorical: mode + one-hot."""

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    )
