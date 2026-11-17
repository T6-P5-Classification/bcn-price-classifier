"""Streamlit app (#15): form logic and end-to-end runs with a synthetic model.

The model here is trained on made-up rows, so these tests check that the app
works and passes the right inputs, not the quality of its predictions. They
need neither the Airbnb snapshot nor the real model file.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from app_logic import bathroom_flags, build_row, form_to_values, predict  # noqa: E402
from src.cleaning import MODEL_FEATURES  # noqa: E402
from src.preprocessing import TARGET_LABELS, build_preprocessor  # noqa: E402
from src.train_app_model import class_labels  # noqa: E402

APP = str(ROOT / "app" / "app.py")
CATEGORICAL = ["neighbourhood_cleansed", "neighbourhood_group_cleansed", "property_type", "room_type"]
HOODS = {"el Raval": "Ciutat Vella", "la Sagrada Família": "Eixample"}
ROOMS = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]


def synthetic_training(n=120, seed=0):
    rng = np.random.default_rng(seed)
    hoods = rng.choice(list(HOODS), n)
    X = pd.DataFrame({
        "neighbourhood_cleansed": hoods,
        "neighbourhood_group_cleansed": [HOODS[h] for h in hoods],
        "latitude": rng.normal(41.39, 0.01, n),
        "longitude": rng.normal(2.17, 0.01, n),
        "property_type": rng.choice(["Entire rental unit", "Room in hotel"], n),
        "room_type": rng.choice(ROOMS, n),
        "accommodates": rng.integers(1, 7, n),
        "bedrooms": rng.integers(0, 4, n).astype(float),
        "beds": rng.integers(1, 5, n).astype(float),
        "minimum_nights": rng.integers(1, 30, n).astype(float),
        "bathrooms_count": rng.integers(1, 3, n).astype(float),
        "is_shared_bathroom": rng.integers(0, 2, n),
        "is_private_bathroom": rng.integers(0, 2, n),
    })
    for column in MODEL_FEATURES:
        if column.startswith("has_"):
            X[column] = rng.integers(0, 2, n)
    X = X[list(MODEL_FEATURES)]
    y = pd.Series(np.tile(TARGET_LABELS, n // 3))
    return X, y


def make_artifact(numeric_labels=False):
    X, y = synthetic_training()
    if numeric_labels:  # like XGBoost: classes 0/1/2
        y = y.map({label: i for i, label in enumerate(TARGET_LABELS)})
    model = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=500))])
    model.fit(X, y)
    by_hood = X.groupby("neighbourhood_cleansed")
    return {
        "model": model,
        "thresholds": {"lower": 122.0, "upper": 250.0},
        "columns": list(X.columns),
        "numeric_columns": list(X.select_dtypes("number").columns),
        "property_types": ["Entire rental unit", "Room in hotel"],
        "room_types": ROOMS,
        "neighbourhoods": sorted(HOODS),
        "hood_to_district": HOODS,
        "hood_coords": by_hood[["latitude", "longitude"]].median().to_dict("index"),
        "model_name": "modelo sintético de test",
        "class_labels": class_labels(model.classes_),
    }


def form(**changes):
    answers = {
        "hood": "el Raval",
        "property_type": "Entire rental unit",
        "room_type": "Entire home/apt",
        "accommodates": 4,
        "bedrooms": 2,
        "beds": 3,
        "minimum_nights": 2,
        "bathrooms_count": 1.5,
        "bathroom": "own",
        "amenities": {c: c == "has_tv" for c in MODEL_FEATURES if c.startswith("has_")},
    }
    answers.update(changes)
    return answers


@pytest.fixture
def run_app(tmp_path, monkeypatch):
    def run(numeric_labels=False):
        path = tmp_path / ("numeric.joblib" if numeric_labels else "text.joblib")
        joblib.dump(make_artifact(numeric_labels), path)
        monkeypatch.setenv("APP_MODEL_PATH", str(path))
        return AppTest.from_file(APP, default_timeout=30).run()
    return run


# --- Form -> model inputs -------------------------------------------------

def test_form_fills_the_23_model_variables_with_the_user_answers():
    artifact = make_artifact()
    values = form_to_values(artifact, form())
    assert set(values) == set(MODEL_FEATURES)
    assert values["neighbourhood_group_cleansed"] == "Ciutat Vella"
    assert values["latitude"] == artifact["hood_coords"]["el Raval"]["latitude"]
    assert (values["accommodates"], values["bedrooms"], values["beds"]) == (4, 2, 3)
    assert (values["minimum_nights"], values["bathrooms_count"]) == (2, 1.5)
    assert values["has_tv"] == 1 and values["has_pool"] == 0
    assert values["room_type"] == "Entire home/apt"  # declared, not derived


@pytest.mark.parametrize("room_type, bathroom, flags", [
    ("Entire home/apt", "own", (0, 0)),
    ("Private room", "private", (0, 1)),
    ("Private room", "shared", (1, 0)),
    ("Hotel room", "private", (0, 1)),
    ("Shared room", "shared", (1, 0)),
])
def test_bathroom_flags_follow_the_answer(room_type, bathroom, flags):
    assert bathroom_flags(room_type, bathroom) == flags


@pytest.mark.parametrize("room_type, bathroom", [
    ("Entire home/apt", "private"),
    ("Entire home/apt", "shared"),
    ("Private room", "own"),
    ("Shared room", "private"),
])
def test_combinations_without_training_examples_are_rejected_not_corrected(room_type, bathroom):
    with pytest.raises(ValueError):
        bathroom_flags(room_type, bathroom)


def test_build_row_rejects_missing_or_empty_variables():
    artifact = make_artifact()
    values = form_to_values(artifact, form())
    with pytest.raises(ValueError, match="missing"):
        build_row(artifact, {k: v for k, v in values.items() if k != "beds"})
    with pytest.raises(ValueError, match="empty"):
        build_row(artifact, {**values, "beds": float("nan")})
    row = build_row(artifact, values)
    assert row.columns.tolist() == list(MODEL_FEATURES) and row.notna().all(axis=None)


# --- Model labels ---------------------------------------------------------

@pytest.mark.parametrize("numeric_labels", [False, True])
def test_prediction_is_shown_with_class_names_for_text_or_numeric_models(numeric_labels):
    artifact = make_artifact(numeric_labels)
    prediction, probabilities = predict(artifact, build_row(artifact, form_to_values(artifact, form())))
    assert prediction in TARGET_LABELS
    assert sorted(probabilities.index) == sorted(TARGET_LABELS)
    assert probabilities.sum() == pytest.approx(1.0)
    assert prediction == probabilities.idxmax()


# --- Whole app (AppTest) --------------------------------------------------

@pytest.mark.parametrize("numeric_labels", [False, True])
def test_app_loads_and_predicts(run_app, numeric_labels):
    at = run_app(numeric_labels)
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert any(label in page for label in TARGET_LABELS)
    assert "Probabilidad estimada por el modelo" in page
    assert "Modelo provisional para demostración" in at.warning[0].value


def test_app_explains_an_inconsistent_room_and_bathroom(run_app):
    at = run_app()
    room =next(s for s in at.selectbox if s.label == "Qué se alquila")
    room.set_value("Private room")  # bathroom left as "own" -> must be asked
    at.button[0].click().run()
    assert not at.exception
    assert "privado o compartido" in at.error[0].value
