"""Shared preprocessing for every model notebook (#13).

``build_preprocessor()`` returns an unfitted ColumnTransformer. It must always
be used inside a Pipeline, so medians, scaling and categories are learned only
from the training data (or from each CV training fold).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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