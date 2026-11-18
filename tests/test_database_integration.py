"""Opt-in integration checks against an isolated LOCAL MongoDB, never Atlas.

Set TEST_MONGODB_URI to a loopback server. Each test owns a randomly named
database and removes only that database; existing application data is untouched.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from threading import Barrier
from urllib.parse import urlsplit, urlunsplit
import uuid

import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pytest
from streamlit.testing.v1 import AppTest

from src.cleaning import MODEL_FEATURES
from src.database import mongo_store
import src.feedback_store as feedback


THRESHOLDS = {"lower": 122.0, "upper": 249.72333333333327}
PROBABILITIES = {"Económico": .2, "Medio": .65, "Premium": .15}


@pytest.fixture
def mongo_backend(monkeypatch):
    uri = os.environ.get("TEST_MONGODB_URI")
    if not uri:
        pytest.skip("Set TEST_MONGODB_URI to an isolated local MongoDB to run integration tests")
    parsed = urlsplit(uri)
    if parsed.scheme != "mongodb" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("Integration tests require a loopback mongodb:// server; remote servers are refused")
    name = "test_issue27_" + uuid.uuid4().hex
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    assert name not in client.list_database_names()
    test_uri = urlunsplit(parsed._replace(path="/" + name))
    monkeypatch.setenv("MONGODB_URI", test_uri)
    store = mongo_store(test_uri)
    try:
        yield client[name], store
    finally:
        store.close()
        mongo_store.cache_clear()
        client.drop_database(name)
        client.close()


def listing():
    values = {column: np.int64(1) for column in MODEL_FEATURES}
    values.update(neighbourhood_cleansed="synthetic-hood", neighbourhood_group_cleansed="synthetic-district",
                  property_type="Entire rental unit", room_type="Entire home/apt",
                  latitude=np.float64(41.0), longitude=np.float64(2.0))
    return values


def save(directory=None):
    return feedback.save_prediction(listing(), "Medio", PROBABILITIES,
                                    model_name="synthetic", model_version="issue27-test",
                                    thresholds=THRESHOLDS, directory=directory)


def test_round_trip_and_csv_contract(mongo_backend, tmp_path):
    db, _ = mongo_backend
    assert list(feedback.load_predictions()) == feedback.PREDICTION_COLUMNS
    assert list(feedback.load_feedback()) == feedback.FEEDBACK_COLUMNS
    assert feedback.load_labelled().empty
    identifier = save()
    csv_identifier = save(tmp_path)
    feedback.save_feedback(identifier, 150.)
    feedback.save_feedback(csv_identifier, 150., directory=tmp_path)
    prediction = feedback.load_predictions()
    answer = feedback.load_feedback()
    assert len(prediction) == len(answer) == len(feedback.load_labelled()) == 1
    assert prediction.iloc[0].prediction_id == identifier
    assert answer.iloc[0].real_class == "Medio" and bool(answer.iloc[0].correct)
    assert "_id" not in prediction and "_id" not in answer
    for mongo_frame, csv_frame in [(prediction, feedback.load_predictions(tmp_path)),
                                    (answer, feedback.load_feedback(tmp_path))]:
        pd.testing.assert_frame_equal(mongo_frame.drop(columns=["prediction_id", "created_at"]),
                                      csv_frame.drop(columns=["prediction_id", "created_at"]), check_dtype=False)
    for table in ("predictions", "feedback"):
        indexes = db[table].index_information()
        assert indexes["prediction_id_unique"]["unique"] is True


def test_unknown_prediction_and_duplicate_feedback_are_rejected(mongo_backend):
    with pytest.raises(ValueError, match="Unknown prediction_id"):
        feedback.save_feedback("unknown-example", 150.)
    identifier = save()
    feedback.save_feedback(identifier, 150.)
    with pytest.raises(ValueError, match="Ya se ha guardado"):
        feedback.save_feedback(identifier, 300.)
    assert len(feedback.load_feedback()) == 1


@pytest.mark.parametrize("price,expected", [(122., "Económico"),
                                           (249.72333333333327, "Medio"), (249.73, "Premium")])
def test_original_thresholds_are_preserved(mongo_backend, price, expected):
    row = feedback.save_feedback(save(), price)
    assert row["real_class"] == expected
    assert feedback.load_predictions().iloc[0].threshold_upper == THRESHOLDS["upper"]


def test_concurrent_feedback_is_enforced_by_the_database(mongo_backend, monkeypatch):
    db, store = mongo_backend
    identifier = save()
    barrier = Barrier(4)
    insert = store.append
    def synchronized_insert(table, row):
        if table == "feedback":
            barrier.wait(timeout=15)
        return insert(table, row)
    monkeypatch.setattr(store, "append", synchronized_insert)
    def answer(price):
        try:
            feedback.save_feedback(identifier, price)
            return "saved"
        except ValueError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(answer, [90., 150., 300., 500.]))
    assert outcomes.count("saved") == 1 and outcomes.count("rejected") == 3
    assert db.feedback.count_documents({"prediction_id": identifier}) == 1


def test_reconnecting_does_not_lose_records(mongo_backend):
    _, store = mongo_backend
    identifier = save()
    feedback.save_feedback(identifier, 150.)
    store.close()
    mongo_store.cache_clear()
    assert feedback.load_labelled().iloc[0].prediction_id == identifier
    # Close the new pooled client too; the fixture closes its original client.
    mongo_store(os.environ["MONGODB_URI"]).close()
    mongo_store.cache_clear()


def test_app_prediction_and_feedback_use_mongodb(mongo_backend, tmp_path, monkeypatch):
    # Reuse the existing synthetic app artifact, not the real reserved dataset.
    from test_app import make_artifact, APP
    db, _ = mongo_backend
    path = tmp_path / "synthetic.joblib"
    joblib.dump(make_artifact(numeric_labels=True), path)
    monkeypatch.setenv("APP_MODEL_PATH", str(path))
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path / "unused-csv"))
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    next(n for n in app.number_input if n.label.startswith("Precio real")).set_value(150.)
    next(b for b in app.button if b.label == "Enviar").click().run()
    assert not app.exception and len(app.success) == 1
    assert db.predictions.count_documents({}) == db.feedback.count_documents({}) == 1
    assert len(feedback.load_labelled()) == 1
    assert not (tmp_path / "unused-csv").exists()
