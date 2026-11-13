"""Train a temporary model for the Streamlit app (#15).

Baseline: LogisticRegression inside the shared pipeline, trained on the 23
agreed variables (``MODEL_FEATURES``). When the champion is chosen (#22/#23),
replace the model here and run the script again: the form does not change,
because every model uses the same 23 variables.

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
from src.preprocessing import build_preprocessor, load_train_test  # noqa: E402

MODEL_PATH = ROOT / "models" / "app_model.joblib"

# Property types offered in the form: those with at least this many train
# listings (the same limit as ``min_frequency`` in the OneHotEncoder). Rarer
# types are grouped as "infrequent" by the encoder anyway.
MIN_PROPERTY_COUNT = 20


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
        # room_type is derived from property_type (no need to ask twice).
        "property_to_room": X_train.groupby("property_type")["room_type"]
        .agg(lambda s: s.mode()[0])
        .loc[property_types]
        .to_dict(),
        "neighbourhoods": sorted(X_train["neighbourhood_cleansed"].unique()),
        "hood_to_district": by_hood["neighbourhood_group_cleansed"]
        .agg(lambda s: s.mode()[0])
        .to_dict(),
        "hood_coords": by_hood[["latitude", "longitude"]].median().to_dict("index"),
        "model_name": "Logistic Regression (provisional)",
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
