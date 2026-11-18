"""Train the model of the Streamlit app (#15, #23).

The app uses the champion chosen in #22: XGBoost with the hyperparameters of
the Optuna trial selected by the agreed rule (``docs/results/xgboost_details.json``),
inside the shared pipeline and fitted on all of train with the 23 variables of
``MODEL_FEATURES``. The test set is not used. ``class_labels`` tells the app
how to read the model's classes (XGBoost predicts 0/1/2).

For the A/B test (#35) the artifact also carries a challenger: LightGBM with
the hyperparameters of its own selected Optuna trial (#21), fitted on the same
train rows. It has no test evaluation: the test set was used once, for the
champion only.

Usage:  python -m src.train_app_model   (or: python src/train_app_model.py)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # works with "python src/train_app_model.py" too

from src.cleaning import MODEL_FEATURES  # noqa: E402
from src.preprocessing import (  # noqa: E402
    RAW_DATA_PATH, TARGET_LABELS, build_preprocessor, load_train_test,
)
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


def challenger_model(results=RESULTS, *, raw_sha256: str | None = None):
    """Unfitted LightGBM challenger (#35), checked against its comparison row.

    Same construction as notebook 07: shared preprocessing and LGBMClassifier
    with the selected parameters plus the fixed ones. If ``raw_sha256`` is
    given, the challenger must have been selected on those same data.
    """
    details = json.loads((results / "lightgbm_details.json").read_text())
    row = pd.read_csv(results / "lightgbm_comparison.csv").query("model == 'LightGBM'").iloc[0]
    if details["provenance"]["raw_sha256"] != row["raw_sha256"]:
        raise ValueError("lightgbm_details.json and lightgbm_comparison.csv use different data")
    if raw_sha256 is not None and row["raw_sha256"] != raw_sha256:
        raise ValueError("The challenger was selected on different data than the champion")
    model = LGBMClassifier(**details["fixed_params"], **details["selected_params"])
    winner = json.loads(row["params"])
    effective = model.get_params()
    mismatch = {k: (effective.get(k), v) for k, v in winner.items() if not _same(effective.get(k), v)}
    if mismatch:
        raise ValueError(f"Challenger parameters differ from the comparison: {mismatch}")
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def _same_params(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)


def _same_thresholds(a: dict, b: dict) -> bool:
    return all(_same(float(a[k]), float(b[k])) for k in ("lower", "upper"))


def champion_provenance(results=RESULTS) -> dict:
    """How the champion was selected: model, parameters, data signature, train size, thresholds."""
    details = json.loads((results / "xgboost_details.json").read_text())
    row = pd.read_csv(results / "xgboost_comparison.csv").query("model == 'XGBoost'").iloc[0]
    if details["provenance"]["raw_sha256"] != row["raw_sha256"]:
        raise ValueError("xgboost_details.json and xgboost_comparison.csv use different data")
    return {
        "model": "XGBoost",
        "selected_trial": details["selected_trial"],
        "params": json.loads(row["params"]),
        "raw_sha256": row["raw_sha256"],
        "train_rows": details["train_rows"],
        "thresholds": details["thresholds"],
    }


def check_training_data(provenance: dict, *, raw_sha256: str, n_train: int, thresholds: dict) -> None:
    """Stop before fitting if the data are not the ones the champion was selected with."""
    problems = []
    if raw_sha256 != provenance["raw_sha256"]:
        problems.append("data signature (raw_sha256)")
    if n_train != provenance["train_rows"]:
        problems.append(f"train size ({n_train} vs {provenance['train_rows']})")
    if not _same_thresholds(thresholds, provenance["thresholds"]):
        problems.append(f"thresholds ({thresholds} vs {provenance['thresholds']})")
    if problems:
        raise ValueError(f"Training data differ from the champion selection: {problems}")


def held_out_summary(provenance: dict, results=RESULTS) -> dict | None:
    """Metrics of the single test evaluation (#22), only if they belong to this model.

    Without a record the app works and shows no test metrics. A record of a
    different model, parameters, data or thresholds is rejected, so that old
    results are never shown as if they validated the current model.
    """
    path = results / "champion_test.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    problems = []
    if record.get("champion") != provenance["model"]:
        problems.append(f"model ({record.get('champion')})")
    if record.get("selected_trial") != provenance["selected_trial"]:
        problems.append("Optuna trial")
    if not _same_params(record.get("params", {}), provenance["params"]):
        problems.append("parameters")
    if record.get("provenance", {}).get("raw_sha256") != provenance["raw_sha256"]:
        problems.append("data signature (raw_sha256)")
    if record.get("n_train") != provenance["train_rows"]:
        problems.append("train size")
    if not _same_thresholds(record.get("thresholds", {"lower": "nan", "upper": "nan"}),
                            provenance["thresholds"]):
        problems.append("thresholds")
    if problems:
        raise ValueError(f"champion_test.json does not belong to this model: {problems}")
    return {"f1_macro": record["test"]["f1_macro"], "accuracy": record["test"]["accuracy"],
            "n_test": record["n_test"]}


def model_version(name: str = "xgboost") -> str:
    """Identifier of the trained artifact stored with every prediction."""
    trial = json.loads((RESULTS / f"{name}_details.json").read_text())["selected_trial"]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "sin-git"
    return f"{name}-trial{trial}-{commit}"


def main() -> None:
    X_train, _, y_train, _, thresholds = load_train_test(return_thresholds=True)
    assert list(X_train.columns) == list(MODEL_FEATURES), "Expected the 23 MODEL_FEATURES"

    provenance = champion_provenance()
    check_training_data(
        provenance,
        raw_sha256=hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest(),
        n_train=len(X_train),
        thresholds=thresholds,
    )
    test_metrics = held_out_summary(provenance)  # checked before the (slower) fit
    model = champion_model()
    model.fit(X_train, encode_target(y_train))  # Económico=0, Medio=1, Premium=2
    challenger = challenger_model(raw_sha256=provenance["raw_sha256"])
    challenger.fit(X_train, y_train)  # text labels, as in notebook 07

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
        "model_version": model_version(),
        "provisional": False,
        "test_metrics": test_metrics,
        "provenance": provenance,
        "class_labels": class_labels(model.classes_),
        # A/B test (#35): replaces the model and its description for the
        # sessions assigned to the challenger (src/ab_test.py).
        "challenger": {
            "model": challenger,
            "model_name": "LightGBM",
            "model_version": model_version("lightgbm"),
            "class_labels": class_labels(challenger.classes_),
            "test_metrics": None,  # the test set was used once, for the champion
            "ab_role": "challenger",
        },
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
