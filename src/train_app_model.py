"""Train a temporary model for the Streamlit app (#15).

Baseline: LogisticRegression inside the shared pipeline. When the champion is
chosen (#22/#23), replace the model here and run the script again.

Usage (from the repo root):  python -m src.train_app_model
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, load_train_test

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "app_model.joblib"


def main() -> None:
    X_train, X_test, y_train, y_test, thresholds = load_train_test(
        return_thresholds=True
    )
    model = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)

    # Data the form needs: options, typical values and neighbourhood centres.
    by_hood = X_train.groupby("neighbourhood_cleansed")
    artifact = {
        "model": model,
        "thresholds": thresholds,
        "columns": list(X_train.columns),
        "numeric_columns": list(X_train.select_dtypes("number").columns),
        "room_types": sorted(X_train["room_type"].unique()),
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
