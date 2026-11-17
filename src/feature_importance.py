"""Train-only feature importance for a fitted model family.

The helpers use permutation importance on validation folds. They receive
training data only and never evaluate the reserved test set.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

SEED = 42
N_REPEATS = 10
SCORING = "f1_macro"
CV_DESCRIPTION = "StratifiedKFold(5, shuffle=True, random_state=42)"

IMPORTANCE_COLUMNS = [
    "model", "feature", "importance_mean", "importance_std",
    "importance_min", "importance_max", "n_folds", "n_repeats",
    "scoring", "cv", "snapshot", "code_commit", "raw_sha256",
]


def permutation_feature_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    *,
    model_name: str,
    n_repeats: int = N_REPEATS,
    random_state: int = SEED,
    provenance: dict | None = None,
) -> pd.DataFrame:
    """Aggregate permutation importance across the common CV folds."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    if provenance is None:
        raise ValueError("provenance is required")
    for key in ("snapshot", "code_commit", "raw_sha256"):
        if key not in provenance:
            raise ValueError(f"provenance is missing {key}")

    records = []
    started = perf_counter()
    for fold_number, (fit_index, valid_index) in enumerate(folds, start=1):
        X_fit = X.iloc[fit_index]
        y_fit = y.iloc[fit_index]
        X_valid = X.iloc[valid_index]
        y_valid = y.iloc[valid_index]

        fitted = clone(pipeline)
        with threadpool_limits(limits=1):
            fitted.fit(X_fit, y_fit)
            result = permutation_importance(
                fitted,
                X_valid,
                y_valid,
                scoring=SCORING,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1,
            )

        for feature, mean, std in zip(
            X.columns, result.importances_mean, result.importances_std
        ):
            records.append(
                {
                    "fold": fold_number,
                    "feature": feature,
                    "importance": float(mean),
                    "importance_std_fold": float(std),
                }
            )

    if not records:
        raise ValueError("No folds were provided")

    frame = pd.DataFrame(records)
    summary = (
        frame.groupby("feature", sort=False, as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_min=("importance", "min"),
            importance_max=("importance", "max"),
            n_folds=("fold", "nunique"),
        )
    )
    summary = summary.sort_values(
        "importance_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)
    summary.insert(0, "model", model_name)
    summary["n_repeats"] = n_repeats
    summary["scoring"] = SCORING
    summary["cv"] = CV_DESCRIPTION
    summary["snapshot"] = provenance["snapshot"]
    summary["code_commit"] = provenance["code_commit"]
    summary["raw_sha256"] = provenance["raw_sha256"]

    if list(summary.columns) != IMPORTANCE_COLUMNS:
        raise ValueError("Incomplete feature-importance table")
    return summary


def importance_row_to_csv(frame: pd.DataFrame, path) -> None:
    """Write only aggregated importance; no rows or predictions."""
    frame.to_csv(path, index=False)