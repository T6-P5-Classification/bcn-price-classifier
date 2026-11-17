"""Train the model of the Streamlit app (#15, #23).

The app uses the champion chosen in #22: XGBoost with the hyperparameters of
the Optuna trial selected by the agreed rule (``docs/results/xgboost_details.json``),
inside the shared pipeline and fitted on all of train with the 23 variables of
``MODEL_FEATURES``. The test set is not used. ``class_labels`` tells the app
how to read the model's classes (XGBoost predicts 0/1/2).

Usage:  python -m src.train_app_model   (or: python src/train_app_model.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # works with "python src/train_app_model.py" too

from src.cleaning import MODEL_FEATURES  # noqa: E402
from src.preprocessing import TARGET_LABELS, load_train_test  # noqa: E402
from src.xgboost_model import encode_target, make_pipeline  # noqa: E402

MODEL_PATH = ROOT / "models" / "app_model.joblib"
RESULTS = ROOT / "docs" / "results"

# Property types offered in the form: those with at least this many train
# listings (the same limit as ``min_frequency`` in the OneHotEncoder). Rarer
# types are grouped as "infrequent" by the encoder anyway.
MIN_PROPERTY_COUNT = 20


def class_labels(classes) -> dict:
    """Model class -> label shown in the app (identity for text labels)."""
    return {c: c if isinstance(c, str) else TARGET_LABELS[int(c)] for c in classes}


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return bool(np.isclose(a, b, rtol=1e-12, atol=0))
    return a == b


def champion_model(results=RESULTS):
    """Unfitted champion pipeline, checked against the winning comparison row."""
    details = json.loads((results / "xgboost_details.json").read_text())
    params = details["selected_params"]
    row = pd.read_csv(results / "xgboost_comparison.csv").query("model == 'XGBoost'").iloc[0]
    winner = json.loads(row["params"])
    model = make_pipeline(params)
    effective = model.named_steps["model"].get_params()
    mismatch = {k: (effective.get(k), v) for k, v in winner.items() if not _same(effective.get(k), v)}
    if mismatch:
        raise ValueError(f"Champion parameters differ from the comparison: {mismatch}")
    return model


def held_out_summary(results=RESULTS) -> dict | None:
    """Metrics of the single test evaluation (#22), if it has been recorded."""
    path = results / "champion_test.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return {"f1_macro": record["test"]["f1_macro"], "accuracy": record["test"]["accuracy"],
            "n_test": record["n_test"]}


def main() -> None:
    X_train, _, y_train, _, thresholds = load_train_test(return_thresholds=True)
    assert list(X_train.columns) == list(MODEL_FEATURES), "Expected the 23 MODEL_FEATURES"

    model = champion_model()
    model.fit(X_train, encode_target(y_train))  # Económico=0, Medio=1, Premium=2

    # Data the form needs: options and values derived from the user's choices.
    by_hood = X_train.groupby("neighbourhood_cleansed")
    property_counts = X_train["property_type"].value_counts()
    property_types = property_counts[property_counts >= MIN_PROPERTY_COUNT].index
    artifact = {
        "model": model,
        "thresholds": thresholds,
        "columns": list(X_train.columns),
        "numeric_columns": list(X_train.select_dtypes("number").columns),
        # Most frequent first, so the default option is the usual one.
        "property_types": list(property_types),
        # Asked in the form, not derived from property_type: the same type can
        # be rented whole or by rooms (e.g. "Room in hotel").
        "room_types": list(X_train["room_type"].value_counts().index),
        # Room types observed with each property type in train: the app rejects
        # the others (e.g. "Entire rental unit" rented as a shared room).
        "room_types_by_property": {
            p: list(group.value_counts().index)
            for p, group in X_train.groupby("property_type")["room_type"]
            if p in set(property_types)
        },
        "neighbourhoods": sorted(X_train["neighbourhood_cleansed"].unique()),
        "hood_to_district": by_hood["neighbourhood_group_cleansed"]
        .agg(lambda s: s.mode()[0])
        .to_dict(),
        "hood_coords": by_hood[["latitude", "longitude"]].median().to_dict("index"),
        "model_name": "XGBoost",
        "provisional": False,
        "test_metrics": held_out_summary(),
        "class_labels": class_labels(model.classes_),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
