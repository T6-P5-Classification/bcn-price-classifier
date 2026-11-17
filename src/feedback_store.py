"""Predictions and feedback of the app: one shared storage interface (#23).

Contract for the issues that use this data (see docs/feedback.md):

- #24 new data for retraining, #34 retraining  -> load_labelled()
- #25 live metrics                             -> load_predictions(), load_labelled()
- #36 data drift                               -> load_predictions() (the 23 inputs)
- #35 A/B testing                              -> "variant" column
- #27 MongoDB                                  -> replace the private helpers
  ``_lock``, ``_append`` and ``_read`` with MongoDB; the public functions,
  columns and ``StorageError`` stay the same, so the code of the other issues
  does not change. MongoDB must keep one feedback per prediction (e.g. a
  unique index on ``feedback.prediction_id``).

Two tables (CSV files now, MongoDB collections later):

- ``predictions``: one row per prediction shown in the app.
- ``feedback``: at most one row per prediction, when the user gives the real
  price per night. It points to its prediction with ``prediction_id``.

No personal data and no free text are stored: only the 23 model inputs of the
listing, the model output and the price the user types.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cleaning import MODEL_FEATURES
from src.preprocessing import TARGET_LABELS, create_price_target

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "feedback"
PREDICTIONS = "predictions"
FEEDBACK = "feedback"

PROBABILITY_COLUMNS = {
    "Económico": "prob_economico",
    "Medio": "prob_medio",
    "Premium": "prob_premium",
}
PREDICTION_COLUMNS = [
    "prediction_id",      # str, uuid4
    "created_at",         # str, ISO 8601 UTC
    "model_name",         # str, e.g. "XGBoost"
    "model_version",      # str, identifies the trained artifact
    "variant",            # str, "champion" (or "challenger" for #35)
    *MODEL_FEATURES,      # the 23 model inputs, same names and meaning as train
    "predicted_class",    # str, Económico / Medio / Premium
    *PROBABILITY_COLUMNS.values(),  # float, sum to 1
    "threshold_lower",    # float, train thresholds used to define the classes
    "threshold_upper",
]
FEEDBACK_COLUMNS = [
    "prediction_id",      # str, the prediction this answer refers to
    "created_at",         # str, ISO 8601 UTC
    "real_price",         # float, € per night typed by the user (> 0)
    "real_class",         # str, class of real_price with the prediction's thresholds
    "correct",            # bool, real_class == predicted_class
]
COLUMNS = {PREDICTIONS: PREDICTION_COLUMNS, FEEDBACK: FEEDBACK_COLUMNS}
LOCK_TIMEOUT = 10.0   # seconds waiting for another writer
STALE_LOCK = 60.0     # a lock older than this was left by a crashed process


class StorageError(Exception):
    """The predictions/feedback storage could not be read or written.

    Common to every backend (CSV now, MongoDB in #27): the app catches it and
    keeps showing the result without confirming a failed save.
    """


def storage_dir(directory: str | Path | None = None) -> Path:
    """Folder of the CSV files: argument, then FEEDBACK_DIR, then data/feedback."""
    return Path(directory or os.environ.get("FEEDBACK_DIR") or DEFAULT_DIR)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Storage backend: the only part #27 replaces with MongoDB ---------------

@contextmanager
def _lock(directory: str | Path | None = None):
    """Exclusive access to the CSV files, also between processes.

    A lock file created with O_EXCL works on macOS, Linux and Windows. Checking
    and writing feedback happen inside it, so two simultaneous answers cannot
    both be saved.
    """
    folder = storage_dir(directory)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StorageError(f"No se puede crear la carpeta de datos: {error}") from error
    lock = folder / ".lock"
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(handle)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > STALE_LOCK:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise StorageError("Los datos están ocupados; inténtalo de nuevo.") from None
            time.sleep(0.01)
        except OSError as error:
            raise StorageError(f"No se puede acceder a los datos: {error}") from error
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _append(table: str, row: dict, directory: str | Path | None = None) -> None:
    path = storage_dir(directory) / f"{table}.csv"
    try:
        pd.DataFrame([row], columns=COLUMNS[table]).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )
    except OSError as error:
        raise StorageError(f"No se ha podido guardar: {error}") from error


def _read(table: str, directory: str | Path | None = None) -> pd.DataFrame:
    path = storage_dir(directory) / f"{table}.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS[table])
    try:
        return pd.read_csv(path, dtype={"prediction_id": str})[COLUMNS[table]]
    except (OSError, ValueError, KeyError) as error:
        raise StorageError(f"No se han podido leer los datos: {error}") from error


# --- Public interface --------------------------------------------------------

def price_class(price: float, thresholds: dict) -> str:
    """Class of a price with the same rule as the training target."""
    if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError("El precio real debe ser un número mayor que 0.")
    return str(create_price_target(pd.Series([float(price)]), thresholds).iloc[0])


def save_prediction(
    values: dict,
    predicted_class: str,
    probabilities: dict,
    *,
    model_name: str,
    model_version: str,
    thresholds: dict,
    variant: str = "champion",
    directory: str | Path | None = None,
) -> str:
    """Store one prediction and return its prediction_id."""
    missing = set(MODEL_FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing model inputs: {sorted(missing)}")
    if predicted_class not in TARGET_LABELS:
        raise ValueError(f"Unknown class: {predicted_class!r}")
    if set(probabilities) != set(TARGET_LABELS):
        raise ValueError("Probabilities must contain the three classes")
    try:  # numpy types (XGBoost returns float32) are converted, text is not accepted
        values_p = [float(probabilities[label]) for label in TARGET_LABELS
                    if not isinstance(probabilities[label], (str, bool))]
    except (TypeError, ValueError):
        values_p = []
    if len(values_p) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values_p):
        raise ValueError(f"Probabilities must be finite numbers between 0 and 1: {probabilities}")
    if not math.isclose(sum(values_p), 1.0, abs_tol=1e-6):
        raise ValueError(f"Probabilities must sum to 1: {probabilities}")
    prediction_id = str(uuid.uuid4())
    row = {
        "prediction_id": prediction_id,
        "created_at": _now(),
        "model_name": model_name,
        "model_version": model_version,
        "variant": variant,
        **{feature: values[feature] for feature in MODEL_FEATURES},
        "predicted_class": predicted_class,
        **dict(zip(PROBABILITY_COLUMNS.values(), values_p)),
        "threshold_lower": float(thresholds["lower"]),
        "threshold_upper": float(thresholds["upper"]),
    }
    with _lock(directory):
        _append(PREDICTIONS, row, directory)
    return prediction_id


def save_feedback(
    prediction_id: str, real_price: float, *, directory: str | Path | None = None
) -> dict:
    """Store the real price of a prediction (once) and return the saved row.

    The check "no answer yet" and the write happen under the same lock, so
    two simultaneous answers for one prediction cannot both be saved.
    """
    with _lock(directory):
        predictions = _read(PREDICTIONS, directory)
        match = predictions.loc[predictions["prediction_id"] == prediction_id]
        if match.empty:
            raise ValueError(f"Unknown prediction_id: {prediction_id}")
        if (_read(FEEDBACK, directory)["prediction_id"] == prediction_id).any():
            raise ValueError("Ya se ha guardado una respuesta para esta predicción.")
        prediction = match.iloc[0]
        thresholds = {"lower": prediction["threshold_lower"], "upper": prediction["threshold_upper"]}
        real_class = price_class(real_price, thresholds)
        row = {
            "prediction_id": prediction_id,
            "created_at": _now(),
            "real_price": float(real_price),
            "real_class": real_class,
            "correct": real_class == prediction["predicted_class"],
        }
        _append(FEEDBACK, row, directory)
    return row


def load_predictions(directory: str | Path | None = None) -> pd.DataFrame:
    """Every stored prediction (for #25 metrics and #36 drift)."""
    return _read(PREDICTIONS, directory)


def load_feedback(directory: str | Path | None = None) -> pd.DataFrame:
    """Every stored answer, without the prediction details."""
    return _read(FEEDBACK, directory)


def load_labelled(directory: str | Path | None = None) -> pd.DataFrame:
    """Predictions that have a real price: inputs, output and real class.

    One row per prediction with feedback (#24 new data, #25 accuracy). The
    feedback ``created_at`` is renamed ``feedback_at``.
    """
    feedback = _read(FEEDBACK, directory).rename(columns={"created_at": "feedback_at"})
    return _read(PREDICTIONS, directory).merge(feedback, on="prediction_id", how="inner")
