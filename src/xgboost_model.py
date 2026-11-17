"""Train-only XGBoost experiment for issue 20.

Data loading remains in preprocessing.py. These helpers accept training data
only; they never receive the reserved test or persist individual predictions.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.preprocessing import TARGET_LABELS, build_preprocessor


SEED = 42
N_TRIALS = 40
GAP_LIMIT = 0.05
CV_DESCRIPTION = "StratifiedKFold(5, shuffle=True, random_state=42)"
LABEL_TO_INT = {label: position for position, label in enumerate(TARGET_LABELS)}
FIXED_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 1,
    "verbosity": 0,
}
SEARCH_SPACE = {
    "n_estimators": {"kind": "int", "low": 100, "high": 500, "step": 50},
    "max_depth": {"kind": "int", "low": 2, "high": 6},
    "learning_rate": {"kind": "float", "low": 0.02, "high": 0.15, "log": True},
    "min_child_weight": {"kind": "float", "low": 5.0, "high": 100.0, "log": True},
    "gamma": {"kind": "float", "low": 0.0, "high": 5.0},
    "subsample": {"kind": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"kind": "float", "low": 0.6, "high": 1.0},
    "reg_alpha": {"kind": "float", "low": 0.001, "high": 10.0, "log": True},
    "reg_lambda": {"kind": "float", "low": 0.1, "high": 100.0, "log": True},
}
COMPARISON_COLUMNS = [
    "model", "cv_train_f1_macro", "cv_validation_f1_macro", "cv_validation_std",
    "gap", "gap_under_005", "cv_fit_seconds", "evaluation_set", "oof_accuracy",
    "oof_f1_macro", "oof_extreme_error_rate", "oof_seconds", "search_seconds",
    "params", "cv", "snapshot", "code_commit", "raw_sha256",
]


def encode_target(y: pd.Series) -> pd.Series:
    """Use the agreed ordinal mapping, not alphabetical LabelEncoder order."""
    encoded = y.astype(object).map(LABEL_TO_INT)
    if encoded.isna().any() or set(encoded.unique()) != {0, 1, 2}:
        raise ValueError("Training target must contain the three agreed price classes")
    return encoded.astype("int64").rename("price_segment")


def make_folds(X: pd.DataFrame, y: pd.Series) -> list:
    """Materialize the common partitions once for all configurations."""
    if not X.index.equals(y.index):
        raise ValueError("Training features and target must have identical index order")
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))


def make_pipeline(params: dict | None = None, *, dummy: bool = False) -> Pipeline:
    """Always return an unfitted full preprocessing + classifier pipeline."""
    if dummy:
        model = DummyClassifier(strategy="most_frequent")
    else:
        params = {} if params is None else params
        if set(params) & set(FIXED_PARAMS):
            raise ValueError("Search parameters must not override the fixed protocol")
        model = XGBClassifier(**FIXED_PARAMS, **params)
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def cv_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: list) -> dict:
    """Score only CV on training rows; sklearn's test_score means fold validation."""
    with threadpool_limits(limits=1):
        scores = cross_validate(
            pipeline, X, y, cv=folds, scoring="f1_macro", return_train_score=True,
            error_score="raise", n_jobs=1,
        )
    train = float(scores["train_score"].mean())
    validation = float(scores["test_score"].mean())
    return {
        "cv_train_f1_macro": train,
        "cv_validation_f1_macro": validation,
        # Population std matches the proposed common CSV contract.
        "cv_validation_std": float(scores["test_score"].std(ddof=0)),
        "gap": train - validation,
        "gap_under_005": bool(train - validation < GAP_LIMIT),
        "cv_fit_seconds": float(scores["fit_time"].sum()),
        "folds": [
            {"fold": i + 1, "f1_train": float(a), "f1_validation": float(b),
             "fit_seconds": float(seconds)}
            for i, (a, b, seconds) in enumerate(zip(
                scores["train_score"], scores["test_score"], scores["fit_time"]
            ))
        ],
    }


def run_search(X: pd.DataFrame, y: pd.Series, folds: list, *, n_trials: int = N_TRIALS):
    """Maximize validation macro F1, recording the gap without consulting test.

    No pruning/early stopping or automatic extra budget. Eligibility is applied
    after the search; the unconstrained Optuna best need not be deliverable.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")

    def objective(trial):
        params = {}
        for name, definition in SEARCH_SPACE.items():
            options = {key: value for key, value in definition.items() if key != "kind"}
            suggest = trial.suggest_int if definition["kind"] == "int" else trial.suggest_float
            params[name] = suggest(name, **options)
        metrics = cv_metrics(make_pipeline(params), X, y, folds)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics["cv_validation_f1_macro"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    started = perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study, perf_counter() - started


def select_trial(study: optuna.Study):
    """Best validation F1 among completed trials with strict gap < 0.05.

    Never silently deliver an ineligible model when the search misses the goal.
    Negative gaps remain visible; they are not clipped to zero.
    """
    eligible = [
        trial for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None and np.isfinite(trial.value)
        and np.isfinite(trial.user_attrs.get("gap", np.nan))
        and trial.user_attrs["gap"] < GAP_LIMIT
    ]
    if not eligible:
        raise ValueError("No completed trial meets gap < 0.05; report unmet criterion")
    # Deterministic tie break: earliest trial wins at exactly the same value.
    return max(eligible, key=lambda trial: (trial.value, -trial.number))


def oof_diagnostics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: list):
    """One held-out prediction per training row; never fit a final full-train model."""
    started = perf_counter()
    with threadpool_limits(limits=1):
        predicted = cross_val_predict(
            clone(pipeline), X, y, cv=folds, method="predict", n_jobs=1
        )
    elapsed = perf_counter() - started
    actual = np.asarray(y)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2])
    extreme = ((actual == 0) & (predicted == 2)) | ((actual == 2) & (predicted == 0))
    return {
        "evaluation_set": "train_oof",
        "oof_accuracy": float(accuracy_score(actual, predicted)),
        "oof_f1_macro": float(f1_score(actual, predicted, average="macro")),
        "oof_extreme_error_rate": float(extreme.mean()),
        "oof_seconds": elapsed,
        "n_rows": len(actual),
        "confusion_matrix": matrix.tolist(),
        "matrix_rows": "real price class",
        "matrix_columns": "predicted class",
        "label_order": list(TARGET_LABELS),
        "classification_report": classification_report(
            actual, predicted, labels=[0, 1, 2], target_names=TARGET_LABELS,
            output_dict=True, zero_division=0,
        ),
    }


def comparison_row(name, metrics, diagnostics, *, params, search_seconds, provenance):
    """Only aggregated metrics/configuration; no rows, prices, IDs or predictions."""
    row = {key: metrics[key] for key in COMPARISON_COLUMNS if key in metrics}
    row.update({key: diagnostics[key] for key in COMPARISON_COLUMNS if key in diagnostics})
    row.update(
        model=name, params=json.dumps(params, sort_keys=True),
        search_seconds=float(search_seconds), cv=CV_DESCRIPTION,
        snapshot=provenance["snapshot"], code_commit=provenance["code_commit"],
        raw_sha256=provenance["raw_sha256"],
    )
    if set(row) != set(COMPARISON_COLUMNS):
        raise ValueError("Incomplete comparison row")
    return row


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """Flatten the search record, including unsuccessful states if any."""
    return pd.DataFrame([
        {
            "number": trial.number, "state": trial.state.name,
            "f1_validation": trial.value,
            "f1_train": trial.user_attrs.get("cv_train_f1_macro"),
            "validation_std": trial.user_attrs.get("cv_validation_std"),
            "gap": trial.user_attrs.get("gap"),
            "gap_under_005": trial.user_attrs.get("gap_under_005"),
            "seconds": trial.duration.total_seconds() if trial.duration else None,
            "params": json.dumps(trial.params, sort_keys=True),
        }
        for trial in study.trials
    ])
