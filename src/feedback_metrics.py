"""Aggregate voluntary app feedback, using the public storage schema (#25)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.feedback_store import price_class
from src.preprocessing import TARGET_LABELS

COHORT_COLUMNS = ["model_name", "model_version", "variant", "threshold_lower", "threshold_upper"]
PREDICTION_FIELDS = ["prediction_id", *COHORT_COLUMNS, "predicted_class"]


def feedback_metrics(predictions: pd.DataFrame, labelled: pd.DataFrame) -> dict:
    """One summary per model/version/variant/threshold cohort, no row data.

    Inputs are load_predictions() and load_labelled(). Labels are voluntary,
    unverified user reports, not an independent evaluation of generalization.
    Missing classes count as zero F1 in the fixed three-class macro average.
    Invalid or duplicate data raises ValueError instead of inflating scores.
    """
    if not set(PREDICTION_FIELDS) <= set(predictions.columns):
        raise ValueError("Faltan campos de predicciones para calcular métricas.")
    if not set(PREDICTION_FIELDS + ["real_price", "real_class"]) <= set(labelled.columns):
        raise ValueError("Faltan campos del feedback para calcular métricas.")
    p = predictions[PREDICTION_FIELDS].copy()
    y = labelled[PREDICTION_FIELDS + ["real_price", "real_class"]].copy()
    for frame in (p, y):
        if frame.prediction_id.duplicated().any():
            raise ValueError("Hay registros duplicados por predicción.")
        for column in ["prediction_id", "model_name", "model_version", "variant"]:
            if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError("Faltan identificadores o metadatos de versión válidos.")
        if not frame.predicted_class.isin(TARGET_LABELS).all():
            raise ValueError("Hay una clase predicha desconocida.")
        try:
            thresholds = frame[["threshold_lower", "threshold_upper"]].astype(float)
        except (ValueError, TypeError):
            raise ValueError("Los umbrales del target no son válidos.") from None
        if not np.isfinite(thresholds.to_numpy()).all() or not (
            (thresholds.threshold_lower > 0) &
            (thresholds.threshold_upper > thresholds.threshold_lower)
        ).all():
            raise ValueError("Los umbrales del target no son válidos.")
        frame[["threshold_lower", "threshold_upper"]] = thresholds
    if not y.prediction_id.isin(p.prediction_id).all():
        raise ValueError("Hay feedback sin una predicción correspondiente.")
    if not y.empty:
        expected = p.set_index("prediction_id").loc[y.prediction_id]
        supplied = y.set_index("prediction_id")[expected.columns]
        if not expected.eq(supplied).to_numpy().all():
            raise ValueError("El feedback no coincide con su predicción o versión.")
        try:
            actual = [price_class(float(row.real_price), {"lower": row.threshold_lower,
                      "upper": row.threshold_upper}) for row in y.itertuples()]
        except (ValueError, TypeError, OverflowError):
            raise ValueError("El feedback contiene un precio no válido.") from None
        if not y.real_class.isin(TARGET_LABELS).all() or list(y.real_class) != actual:
            raise ValueError("La clase real no coincide con el precio y sus umbrales.")
    summaries = {}
    labels = list(TARGET_LABELS)
    for cohort, rows in p.groupby(COHORT_COLUMNS, sort=True, dropna=False):
        answers = y[y.prediction_id.isin(rows.prediction_id)]
        n, m = len(rows), len(answers)
        matrix = confusion_matrix(answers.real_class.tolist(), answers.predicted_class.tolist(), labels=labels) if m else np.zeros((3, 3), dtype=int)
        report = classification_report(answers.real_class.tolist(), answers.predicted_class.tolist(), labels=labels, output_dict=True, zero_division=0) if m else {}
        summaries[cohort] = {
            "n_predictions": n, "n_feedback": m, "coverage": m / n,
            "accuracy": float(accuracy_score(answers.real_class, answers.predicted_class)) if m else None,
            "f1_macro": float(f1_score(answers.real_class, answers.predicted_class, labels=labels, average="macro", zero_division=0)) if m else None,
            "extreme_error_rate": float((matrix[0, 2] + matrix[2, 0]) / m) if m else None,
            "confusion_matrix": matrix.tolist(),
            "per_class": {label: report[label] for label in labels} if m else {},
            "prediction_counts": {label: int((rows.predicted_class == label).sum()) for label in labels},
        }
    return summaries
