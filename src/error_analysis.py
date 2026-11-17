"""Issue 17: fixed-model OOF errors, with explicit real/predicted directions.

No test input, hyperparameter search, winner selection or listing-level export.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.preprocessing import TARGET_LABELS, build_preprocessor


MODELS = ("LogisticRegression", "RandomForest", "XGBoost", "LightGBM")
CV_DESCRIPTION = "StratifiedKFold(5, shuffle=True, random_state=42)"
ERROR_COLORS = ("#D89022", "#B4374C")


def selected_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Require exactly the four submitted candidates; ignore only explicit Dummy rows."""
    required = {
        "model", "params", "source_csv", "evaluation_set", "cv", "snapshot",
        "raw_sha256", "gap", "cv_validation_f1_macro", "oof_accuracy", "oof_f1_macro", "oof_extreme_error_rate",
    }
    if not required <= set(results):
        raise ValueError(f"Missing comparison fields: {sorted(required - set(results))}")
    candidates = results.loc[results.model != "DummyClassifier"].copy()
    if set(candidates.model) != set(MODELS) or candidates.model.duplicated().any():
        raise ValueError("Need one selected configuration for each of the four agreed models")
    if not candidates.evaluation_set.eq("train_oof").all() or not candidates.cv.eq(CV_DESCRIPTION).all():
        raise ValueError("Candidates must use train_oof and the agreed five-fold protocol")
    for column in ["snapshot", "raw_sha256"]:
        if candidates[column].isna().any() or candidates[column].nunique() != 1:
            raise ValueError(f"Candidates disagree about {column}")
    if not np.isfinite(candidates.gap).all() or not candidates.gap.lt(0.05).all():
        raise ValueError("A submitted candidate does not meet the agreed mean gap < 0.05")
    for value in candidates.params:
        if not isinstance(json.loads(value), dict):
            raise ValueError("Candidate params must encode a JSON object")
    # Display order is stable, not a new ranking or champion decision.
    return candidates.set_index("model").loc[list(MODELS)].reset_index()


def fixed_pipeline(name: str, params: dict) -> Pipeline:
    """Reconstruct the submitted estimator; never call Optuna or select another trial."""
    if name == "LogisticRegression":
        # The baseline CSV stores C/class_weight; its notebook fixes these defaults.
        estimator = LogisticRegression(**{"solver": "lbfgs", "max_iter": 5000, **params})
    elif name == "RandomForest":
        estimator = RandomForestClassifier(**params)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(**params)
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unsupported candidate: {name}")
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def common_folds(X: pd.DataFrame, y: pd.Series) -> list:
    if X.empty or not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X and y must be nonempty and aligned with unique indices")
    _class_names(y)
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def _class_names(values) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or not len(labels) or not pd.Series(labels).isin(TARGET_LABELS).all():
        raise ValueError("Expected nonempty vectors of the three agreed class names")
    return labels


def analyze_matrix(matrix) -> tuple[dict, pd.DataFrame]:
    """Aggregate six error directions; percentages never conceal their denominator."""
    matrix = np.asarray(matrix)
    if (matrix.shape != (3, 3) or not np.issubdtype(matrix.dtype, np.number)
            or not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.equal(matrix, np.floor(matrix)).all() or matrix.sum() <= 0):
        raise ValueError("Expected a nonnegative integer 3x3 confusion matrix with observations")
    matrix = matrix.astype("int64")
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    errors = total - correct
    extreme = int(matrix[0, 2] + matrix[2, 0])
    neighbour = errors - extreme
    summary = {
        "n_total": total, "n_correct": correct, "n_errors": errors,
        "n_neighbour_errors": neighbour, "n_extreme_errors": extreme,
        "pct_correct_of_all": 100 * correct / total,
        "pct_errors_of_all": 100 * errors / total,
        "pct_neighbour_of_all": 100 * neighbour / total,
        "pct_extreme_of_all": 100 * extreme / total,
        "pct_neighbour_of_errors": 100 * neighbour / errors if errors else np.nan,
        "pct_extreme_of_errors": 100 * extreme / errors if errors else np.nan,
    }
    directions = []
    for real, real_label in enumerate(TARGET_LABELS):
        for predicted, predicted_label in enumerate(TARGET_LABELS):
            if real == predicted:
                continue
            count = int(matrix[real, predicted])
            real_total = int(matrix[real].sum())
            directions.append({
                "real_class": real_label, "predicted_class": predicted_label,
                "error_type": "extremo" if abs(real - predicted) == 2 else "vecino",
                "n_errors": count, "n_real_class": real_total,
                "pct_of_all": 100 * count / total,
                "pct_of_errors": 100 * count / errors if errors else np.nan,
                "pct_of_real_class": 100 * count / real_total if real_total else np.nan,
            })
    return summary, pd.DataFrame(directions)


def analyze_predictions(actual, predicted):
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series) and not actual.index.equals(predicted.index):
        raise ValueError("Real and predicted rows must be in identical index order")
    actual, predicted = _class_names(actual), _class_names(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("Real and predicted vectors must have equal lengths")
    matrix = confusion_matrix(actual, predicted, labels=list(TARGET_LABELS))
    summary, directions = analyze_matrix(matrix)
    return matrix, summary, directions


def reproduce_candidate(
    row: pd.Series, X: pd.DataFrame, y: pd.Series, folds: list,
    *, allow_metric_mismatch: bool = False,
):
    """Five fixed-model fits; return aggregates only, never individual predictions.

    Original Random Forest and XGBoost use the agreed integer mapping; the
    other two estimators retain their original string target representation.
    """
    expected = common_folds(X, y)  # validates alignment/labels without fitting
    if len(folds) != len(expected) or any(
        not np.array_equal(a, c) or not np.array_equal(b, d)
        for (a, b), (c, d) in zip(folds, expected)
    ):
        raise ValueError("Use the exact agreed common folds")
    model_name = row["model"]
    pipeline = fixed_pipeline(model_name, json.loads(row["params"]))
    numeric_target = model_name in {"RandomForest", "XGBoost"}
    mapping = {label: index for index, label in enumerate(TARGET_LABELS)}
    target = y.astype(object).map(mapping).astype("int64") if numeric_target else y
    started = perf_counter()
    with threadpool_limits(limits=1):
        predicted = cross_val_predict(pipeline, X, target, cv=folds, n_jobs=1, method="predict")
    seconds = perf_counter() - started
    if numeric_target:
        if not np.isin(predicted, [0, 1, 2]).all():
            raise ValueError("Estimator returned unknown numeric class labels")
        predicted = np.asarray(TARGET_LABELS)[predicted.astype(int)]
    matrix, summary, directions = analyze_predictions(y, predicted)
    metrics = {
        "oof_accuracy": summary["n_correct"] / summary["n_total"],
        "oof_f1_macro": float(f1_score(y, predicted, average="macro")),
        "oof_extreme_error_rate": summary["n_extreme_errors"] / summary["n_total"],
    }
    source_check = {
        key: {"source": float(row[key]), "local": observed,
              "local_minus_source": observed - float(row[key]),
              "matches": bool(np.isclose(observed, float(row[key]), rtol=0, atol=1e-9))}
        for key, observed in metrics.items()
    }
    for key, check in source_check.items():
        if not check["matches"] and not allow_metric_mismatch:
            raise ValueError(
                f"{model_name}: reproduced {key}={check['local']} differs from saved {row[key]}; "
                "check data, versions and configuration, not test or a new search"
            )
    return {
        "matrix": matrix, "summary": summary, "directions": directions,
        "oof_metrics": metrics, "seconds": seconds,
        "source_metric_check": source_check,
        "oof_metrics_match_source": all(check["matches"] for check in source_check.values()),
        "metric_mismatch_explicitly_allowed": allow_metric_mismatch,
        "effective_params": pipeline["model"].get_params(),
        "target_encoding": "Económico=0, Medio=1, Premium=2" if numeric_target else "original class names",
    }


def row_percentages(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix * 100, totals, out=np.full_like(matrix, np.nan), where=totals != 0)


def plot_matrices(name: str, matrix):
    import matplotlib.pyplot as plt

    percentages = row_percentages(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, data, title, units, limit in zip(
        axes, [np.asarray(matrix), percentages],
        ["Recuentos", "Porcentajes dentro de cada categoría REAL"],
        ["Número de anuncios", "% dentro de la fila real"], [np.max(matrix), 100],
    ):
        image = ax.imshow(np.ma.masked_invalid(data), cmap=plt.get_cmap("Blues").with_extremes(bad="#D4D4D4"), vmin=0, vmax=limit)
        for real in range(3):
            for predicted in range(3):
                value = data[real, predicted]
                text = "Sin datos" if not np.isfinite(value) else (f"{value:.1f} %" if ax is axes[1] else str(int(value)))
                ax.text(predicted, real, text, ha="center", va="center", color="white" if value > limit / 2 else "black")
        ax.set_xticks(range(3), TARGET_LABELS)
        ax.set_yticks(range(3), TARGET_LABELS)
        ax.set(xlabel="Categoría PREDICHA", ylabel="Categoría REAL según precio y umbrales", title=title)
        fig.colorbar(image, ax=ax, shrink=0.7, label=units)
    fig.suptitle(f"{name} · predicciones fuera de fold sobre train", fontsize=14)
    return fig


def plot_error_comparison(summary: pd.DataFrame, *, within_errors: bool = False):
    import matplotlib.pyplot as plt

    if within_errors:
        columns = ["pct_neighbour_of_errors", "pct_extreme_of_errors"]
        xlabel = "Porcentaje de las predicciones EQUIVOCADAS de cada modelo"
        title = "Qué tipo de errores comete cada modelo (excluye aciertos)"
    else:
        columns = ["pct_neighbour_of_all", "pct_extreme_of_all"]
        xlabel = "Porcentaje de TODOS los anuncios evaluados de entrenamiento"
        title = "Frecuencia de errores: las barras no incluyen los aciertos"
    table = summary.set_index("model")[columns].copy()
    table.columns = ["Errores vecinos", "Errores extremos"]
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    table.plot.barh(stacked=True, color=ERROR_COLORS, ax=ax)
    ax.set(xlim=(0, 100), xlabel=xlabel, ylabel="Configuración seleccionada de cada modelo", title=title)
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    for position, (_, row) in enumerate(summary.iterrows()):
        note = "Sin errores" if row.n_errors == 0 else f"n={int(row.n_errors)} errores / {int(row.n_total)} anuncios"
        ax.text(1, position, note, ha="left", va="center", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    return fig


def json_safe(value):
    """Undefined percentages become JSON null, not fabricated zeros or NaN tokens."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value
