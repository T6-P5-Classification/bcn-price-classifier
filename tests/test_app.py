"""Streamlit app (#15): form logic and end-to-end runs with a synthetic model.

The model here is trained on made-up rows, so these tests check that the app
works and passes the right inputs, not the quality of its predictions. They
need neither the Airbnb snapshot nor the real model file.
"""

import json
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

from app_logic import (  # noqa: E402
    bathroom_flags, build_row, check_room_type, customer_notice, form_to_values, model_details, predict,
)
from src.cleaning import MODEL_FEATURES  # noqa: E402
from src.preprocessing import TARGET_LABELS, build_preprocessor  # noqa: E402
from src.train_app_model import (  # noqa: E402
    champion_model, champion_provenance, check_training_data, class_labels, held_out_summary,
)
from src.xgboost_model import FIXED_PARAMS  # noqa: E402

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
        "room_types_by_property": {
            "Entire rental unit": ["Entire home/apt"],
            "Room in hotel": ["Private room", "Hotel room", "Entire home/apt"],
        },
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
    def run(numeric_labels=False, **extra):
        path = tmp_path / ("numeric.joblib" if numeric_labels else "text.joblib")
        joblib.dump({**make_artifact(numeric_labels), **extra}, path)
        monkeypatch.setenv("APP_MODEL_PATH", str(path))
        monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path / "feedback"))
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


def test_room_type_must_match_the_property_type():
    artifact = make_artifact()
    with pytest.raises(ValueError, match="no encaja"):
        form_to_values(artifact, form(room_type="Shared room", bathroom="shared"))
    # Ambiguous types keep every option observed in train, none imposed.
    for room in ["Private room", "Hotel room", "Entire home/apt"]:
        check_room_type(artifact, "Room in hotel", room)


def test_build_row_rejects_missing_or_empty_variables():
    artifact = make_artifact()
    values = form_to_values(artifact, form())
    with pytest.raises(ValueError, match="missing"):
        build_row(artifact, {k: v for k, v in values.items() if k != "beds"})
    with pytest.raises(ValueError, match="empty"):
        build_row(artifact, {**values, "beds": float("nan")})
    with pytest.raises(ValueError, match="empty"):
        build_row(artifact, {**values, "beds": np.float32("nan")})
    with pytest.raises(ValueError, match="empty"):
        build_row(artifact, {**values, "room_type": None})
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
    captions = " ".join(c.value for c in at.caption)
    assert "No es una recomendación de precio" in captions and "provisional" in captions
    assert "Modelo provisional para demostración" in at.expander[0].markdown[0].value


def test_app_rejects_entire_apartment_rented_as_shared_room(run_app):
    """Case reported in review: no prediction for contradictory answers."""
    at = run_app()
    next(s for s in at.selectbox if s.label == "Tipo de alojamiento").set_value("Entire rental unit")
    next(s for s in at.selectbox if s.label == "Qué se alquila").set_value("Shared room")
    at.radio[0].set_value("shared")
    at.button[0].click().run()
    assert not at.exception
    assert "no encaja" in at.error[0].value
    assert not any("Probabilidad estimada" in m.value for m in at.markdown)


def test_app_explains_an_inconsistent_room_and_bathroom(run_app):
    at = run_app()
    next(s for s in at.selectbox if s.label == "Tipo de alojamiento").set_value("Room in hotel")
    next(s for s in at.selectbox if s.label == "Qué se alquila").set_value("Private room")
    # bathroom left as "own" -> the user must say private or shared
    at.button[0].click().run()
    assert not at.exception
    assert "privado o compartido" in at.error[0].value


# --- Champion (#23) -------------------------------------------------------

def test_champion_model_uses_the_selected_xgboost_parameters():
    details = json.loads((ROOT / "docs/results/xgboost_details.json").read_text())
    params = champion_model().named_steps["model"].get_params()
    for key, value in {**FIXED_PARAMS, **details["selected_params"]}.items():
        assert params[key] == pytest.approx(value) if isinstance(value, float) else params[key] == value


def test_champion_model_stops_if_parameters_differ_from_the_comparison(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    source = ROOT / "docs/results"
    details = json.loads((source / "xgboost_details.json").read_text())
    details["selected_params"]["max_depth"] = 2
    (results / "xgboost_details.json").write_text(json.dumps(details))
    (results / "xgboost_comparison.csv").write_text((source / "xgboost_comparison.csv").read_text())
    with pytest.raises(ValueError, match="differ"):
        champion_model(results)


PROVENANCE = {
    "model": "XGBoost", "selected_trial": 34, "params": {"max_depth": 6, "learning_rate": 0.136},
    "raw_sha256": "a" * 64, "train_rows": 10088, "thresholds": {"lower": 122.0, "upper": 249.72},
}


def make_record(**changes):
    record = {
        "champion": "XGBoost", "selected_trial": 34,
        "params": {"max_depth": 6, "learning_rate": 0.136},
        "provenance": {"raw_sha256": "a" * 64}, "n_train": 10088,
        "thresholds": {"lower": 122.0, "upper": 249.72},
        "test": {"f1_macro": 0.7651, "accuracy": 0.7645}, "n_test": 2522,
    }
    record.update(changes)
    return record


def test_champion_provenance_matches_the_comparison_files():
    provenance = champion_provenance()
    details = json.loads((ROOT / "docs/results/xgboost_details.json").read_text())
    assert provenance["model"] == "XGBoost" and provenance["selected_trial"] == details["selected_trial"]
    assert provenance["train_rows"] == details["train_rows"]
    assert provenance["thresholds"] == details["thresholds"]
    assert provenance["params"]["max_depth"] == details["selected_params"]["max_depth"]


@pytest.mark.parametrize("change, message", [
    ({"raw_sha256": "b" * 64}, "data signature"),
    ({"n_train": 999}, "train size"),
    ({"thresholds": {"lower": 10.0, "upper": 20.0}}, "thresholds"),
])
def test_training_stops_if_data_differ_from_the_champion_selection(change, message):
    current = {"raw_sha256": "a" * 64, "n_train": 10088, "thresholds": {"lower": 122.0, "upper": 249.72}}
    check_training_data(PROVENANCE, **current)  # same data: no error
    with pytest.raises(ValueError, match=message):
        check_training_data(PROVENANCE, **{**current, **change})


def test_held_out_summary_is_optional_and_reads_a_matching_record(tmp_path):
    assert held_out_summary(PROVENANCE, tmp_path) is None
    (tmp_path / "champion_test.json").write_text(json.dumps(make_record()))
    assert held_out_summary(PROVENANCE, tmp_path) == {"f1_macro": 0.7651, "accuracy": 0.7645, "n_test": 2522}


@pytest.mark.parametrize("change, message", [
    ({"champion": "RandomForest"}, "model"),
    ({"selected_trial": 12}, "trial"),
    ({"params": {"max_depth": 9, "learning_rate": 0.136}}, "parameters"),
    ({"provenance": {"raw_sha256": "b" * 64}}, "data signature"),
    ({"n_train": 500}, "train size"),
    ({"thresholds": {"lower": 10.0, "upper": 20.0}}, "thresholds"),
])
def test_test_metrics_of_another_model_or_data_are_rejected(tmp_path, change, message):
    """Review case: a Random Forest record with other thresholds must not be shown."""
    (tmp_path / "champion_test.json").write_text(json.dumps(make_record(**change)))
    with pytest.raises(ValueError, match=message):
        held_out_summary(PROVENANCE, tmp_path)


CHAMPION = {"model_name": "XGBoost", "provisional": False,
            "test_metrics": {"f1_macro": 0.7651, "accuracy": 0.7645, "n_test": 2522}}


def test_customer_notice_is_short_and_plain():
    text = customer_notice(CHAMPION)
    assert text.startswith("Estimación orientativa")
    assert "No es una recomendación de precio" in text and "puede no acertar" in text
    for jargon in ["F1", "XGBoost", "macro", "modelos comparados"]:
        assert jargon not in text  # technical details go in «Sobre el modelo»
    assert "provisional" in customer_notice({"model_name": "LogReg"})


def test_model_details_distinguish_provisional_and_champion_models():
    assert model_details({"model_name": "LogReg"}).startswith("Modelo provisional")
    text = model_details(CHAMPION)
    assert "provisional" not in text and "XGBoost" in text
    assert "2.522 anuncios" in text and "76 %" in text and "0,765" in text
    assert "no recomienda un precio" in text  # limits are always available
    assert "provisional" not in model_details({**CHAMPION, "test_metrics": None})


def test_app_shows_the_champion_notice(run_app):
    at = run_app(numeric_labels=True, model_name="XGBoost", provisional=False,
                 test_metrics={"f1_macro": 0.7651, "accuracy": 0.7645, "n_test": 2522})
    at.button[0].click().run()
    assert not at.exception
    assert "Modelo XGBoost" in at.expander[0].markdown[0].value
    assert at.expander[0].label == "ℹ️ Sobre el modelo"
    assert not any("provisional" in c.value for c in at.caption)
    assert not at.warning  # no technical warning box for the host


# --- Feedback (#23) -------------------------------------------------------

def test_each_prediction_is_stored_and_feedback_is_linked(run_app, tmp_path):
    from src.feedback_store import load_labelled, load_predictions
    at = run_app()
    at.button[0].click().run()
    assert not at.exception
    stored = load_predictions(tmp_path / "feedback")
    assert len(stored) == 1 and stored["predicted_class"].iloc[0] in TARGET_LABELS

    price = next(n for n in at.number_input if n.label.startswith("Precio real"))
    price.set_value(300.0)
    next(b for b in at.button if b.label == "Enviar").click().run()
    assert not at.exception
    assert "Premium" in at.success[0].value  # 300 € is Premium with the test thresholds
    labelled = load_labelled(tmp_path / "feedback")
    assert len(labelled) == 1 and labelled["real_class"].iloc[0] == "Premium"
    # The result stays on screen after sending feedback.
    assert any("Probabilidad estimada" in m.value for m in at.markdown)


def test_feedback_without_price_shows_an_error_and_saves_nothing(run_app, tmp_path):
    from src.feedback_store import load_feedback
    at = run_app()
    at.button[0].click().run()
    next(b for b in at.button if b.label == "Enviar").click().run()
    assert not at.exception
    assert "mayor que 0" in at.error[0].value
    assert load_feedback(tmp_path / "feedback").empty


def test_storage_failure_keeps_the_result_and_does_not_confirm(run_app, tmp_path, monkeypatch):
    """Review case: a write error while saving feedback must not crash the app."""
    import src.feedback_store as store
    at = run_app()
    at.button[0].click().run()

    def broken_append(*args, **kwargs):
        raise store.StorageError("disco lleno")

    monkeypatch.setattr(store, "_append", broken_append)
    next(n for n in at.number_input if n.label.startswith("Precio real")).set_value(150.0)
    next(b for b in at.button if b.label == "Enviar").click().run()
    assert not at.exception
    assert "No hemos podido guardar" in at.warning[0].value
    assert not at.success
    assert any("Probabilidad estimada" in m.value for m in at.markdown)


def test_prediction_is_shown_even_if_it_cannot_be_stored(run_app, monkeypatch):
    import src.feedback_store as store
    monkeypatch.setattr(store, "_append", lambda *a, **k: (_ for _ in ()).throw(store.StorageError("x")))
    at = run_app()
    at.button[0].click().run()
    assert not at.exception
    assert any("Probabilidad estimada" in m.value for m in at.markdown)
    assert not any(b.label == "Enviar" for b in at.button)  # no feedback without a stored prediction


def test_csv_fallback_is_visible_before_predicting(run_app):
    at = run_app()
    assert any("CSV local" in message.value and "MongoDB no está configurado" in message.value for message in at.info)


def test_mongodb_configuration_is_visible_and_failure_keeps_prediction(run_app, monkeypatch):
    import src.feedback_store as store
    at = run_app()
    monkeypatch.setenv("MONGODB_URI", "mongodb://unused/example")
    def unavailable(*_):
        raise store.StorageError("MongoDB unavailable")
    monkeypatch.setattr(store, "mongo_store", unavailable)
    at.run()
    assert not any("CSV local" in message.value for message in at.info)
    assert any("Almacenamiento configurado: MongoDB" in c.value for c in at.caption)
    at.button[0].click().run()
    assert not at.exception
    assert any("Probabilidad estimada" in m.value for m in at.markdown)
    assert any("No se ha podido guardar" in c.value for c in at.caption)
    assert not any(b.label == "Enviar" for b in at.button)


# --- A/B test (#35) -------------------------------------------------------

def make_challenger():
    X, y = synthetic_training(seed=1)
    model = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(C=0.1, max_iter=500))])
    model.fit(X, y)
    return {"model": model, "model_name": "LightGBM", "model_version": "lightgbm-test",
            "class_labels": class_labels(model.classes_), "test_metrics": None,
            "ab_role": "challenger"}


AB_CHAMPION = {"model_name": "XGBoost", "model_version": "xgboost-test", "provisional": False}


@pytest.mark.parametrize("split, expected", [("1", "LightGBM"), ("0", "XGBoost")])
def test_ab_split_decides_which_model_answers_and_is_recorded(run_app, tmp_path, monkeypatch,
                                                              split, expected):
    from src.feedback_store import load_predictions
    monkeypatch.setenv("AB_SPLIT", split)
    at = run_app(**AB_CHAMPION, challenger=make_challenger())
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["variant"] == ("challenger" if split == "1" else "champion")
    stored = load_predictions(tmp_path / "feedback")
    assert stored["model_name"].tolist() == [expected]
    assert stored["variant"].tolist() == [at.session_state["variant"]]
    assert f"Modelo {expected}" in at.expander[0].markdown[0].value


def test_ab_variant_stays_the_same_during_a_session(run_app, tmp_path, monkeypatch):
    """Sticky: several predictions in one session use the same model."""
    from src.feedback_store import load_predictions
    monkeypatch.setenv("AB_SPLIT", "0.5")
    at = run_app(**AB_CHAMPION, challenger=make_challenger())
    variant = at.session_state["variant"]
    for beds in (1, 2, 3):
        next(n for n in at.number_input if n.label == "Camas").set_value(beds)
        at.button[0].click().run()
        assert not at.exception and at.session_state["variant"] == variant
    names = load_predictions(tmp_path / "feedback")["model_name"]
    assert len(names) == 3 and names.nunique() == 1


def test_challenger_details_do_not_claim_it_was_chosen_or_tested(run_app, monkeypatch):
    monkeypatch.setenv("AB_SPLIT", "1")
    at = run_app(**AB_CHAMPION, challenger=make_challenger(),
                 test_metrics={"f1_macro": 0.7651, "accuracy": 0.7645, "n_test": 2522})
    at.button[0].click().run()
    details = at.expander[0].markdown[0].value
    assert "en prueba frente al modelo elegido" in details
    assert "elegido entre cuatro" not in details
    assert "2.522" not in details  # the champion's test result is not the challenger's


def test_without_challenger_or_with_a_bad_split_only_the_champion_answers(run_app, tmp_path,
                                                                          monkeypatch):
    from src.feedback_store import load_predictions
    monkeypatch.setenv("AB_SPLIT", "1")
    at = run_app(**AB_CHAMPION)  # artifact without challenger
    at.button[0].click().run()
    assert not at.exception and at.session_state["variant"] == "champion"

    monkeypatch.setenv("AB_SPLIT", "mitad")
    at = run_app(**AB_CHAMPION, challenger=make_challenger())
    at.button[0].click().run()
    assert not at.exception and at.session_state["variant"] == "champion"
    assert set(load_predictions(tmp_path / "feedback")["model_name"]) == {"XGBoost"}
