"""Train-only feature importance for a fitted model family.

The helpers use permutation importance on validation folds. They receive
training data only and never evaluate the reserved test set.
"""

from __future__ import annotations

import json
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


def validate_champion_candidate(
    details: dict,
    comparison: pd.DataFrame,
    X: pd.DataFrame,
    thresholds: dict,
    raw_sha256: str,
    fixed_params: dict | None = None,
) -> tuple[int, dict, dict]:
    """Validate that the champion metadata, CSV and data agree before importance."""
    if not isinstance(details, dict):
        raise ValueError("details must be a JSON object")
    if details.get("issue") != 20:
        raise ValueError("details do not describe the XGBoost candidate")
    if details.get("test_evaluated") is not False:
        raise ValueError("candidate must not have evaluated test")
    if details.get("nested_cv") is not False:
        raise ValueError("candidate metadata is inconsistent")
    if details.get("train_rows") != len(X):
        raise ValueError("training size does not match candidate metadata")
    if list(X.columns) != list(details.get("input_schema", {})):
        raise ValueError("training columns do not match candidate metadata")
    if thresholds.get("lower") != details.get("thresholds", {}).get("lower"):
        raise ValueError("lower threshold does not match candidate metadata")
    if thresholds.get("upper") != details.get("thresholds", {}).get("upper"):
        raise ValueError("upper threshold does not match candidate metadata")
    if raw_sha256 != details.get("provenance", {}).get("raw_sha256"):
        raise ValueError("raw data hash does not match candidate metadata")

    candidate_rows = comparison.loc[comparison["model"].eq("XGBoost")]
    if len(candidate_rows) != 1:
        raise ValueError("expected exactly one XGBoost candidate row")
    candidate_params = json.loads(candidate_rows.iloc[0]["params"])
    selected_params = details.get("selected_params")
    if not isinstance(selected_params, dict):
        raise ValueError("selected_params must be an object")
    expected_params = dict(selected_params)
    if fixed_params is not None:
        expected_params = {**fixed_params, **selected_params}
    if candidate_params != expected_params:
        raise ValueError("candidate CSV differs from selected parameters")

    selected_trial = details.get("selected_trial")
    if not isinstance(selected_trial, int):
        raise ValueError("selected_trial must be an integer")
    return selected_trial, selected_params, candidate_params


def validate_effective_params(pipeline, candidate_params: dict) -> None:
    """Ensure every parameter recorded in the candidate CSV is effective."""
    effective = {key: pipeline["model"].get_params()[key] for key in candidate_params}
    if effective != candidate_params:
        raise ValueError(
            "effective pipeline parameters differ from candidate CSV: "
            f"{effective} != {candidate_params}"
        )


def importance_row_to_csv(frame: pd.DataFrame, path) -> None:
    """Write only aggregated importance; no rows or predictions."""
    frame.to_csv(path, index=False)