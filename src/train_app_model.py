"""Train a temporary model for the Streamlit app (#15).

Baseline: LogisticRegression inside the shared pipeline, trained on the 23
agreed variables (``MODEL_FEATURES``). When the champion is chosen (#22/#23),
replace the model here and run the script again. The form does not change,
because every model uses the same 23 variables, and ``class_labels`` tells
the app how to read the model's classes: text labels (logistic regression)
or numbers 0/1/2 (XGBoost, trained with ``LABEL_TO_INT``).

Usage:  python -m src.train_app_model   (or: python src/train_app_model.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # works with "python src/train_app_model.py" too

from src.cleaning import MODEL_FEATURES  # noqa: E402
from src.preprocessing import TARGET_LABELS, build_preprocessor, load_train_test  # noqa: E402

MODEL_PATH = ROOT / "models" / "app_model.joblib"

# Property types offered in the form: those with at least this many train
# listings (the same limit as ``min_frequency`` in the OneHotEncoder). Rarer
# types are grouped as "infrequent" by the encoder anyway.
MIN_PROPERTY_COUNT = 20


def class_labels(classes) -> dict:
    """Model class -> label shown in the app (identity for text labels)."""
    return {c: c if isinstance(c, str) else TARGET_LABELS[int(c)] for c in classes}


def main() -> None:
    X_train, X_test, y_train, y_test, thresholds = load_train_test(
        return_thresholds=True
    )
    assert list(X_train.columns) == list(MODEL_FEATURES), "Expected the 23 MODEL_FEATURES"

    model = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)

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
        "model_name": "Regresión logística provisional",
        "class_labels": class_labels(model.classes_),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
