"""Driver-level checks using mocks, with no MongoDB or credentials required."""

from unittest.mock import MagicMock
import traceback

import numpy as np
import pandas as pd
import pytest
from pymongo.errors import DuplicateKeyError, OperationFailure, ServerSelectionTimeoutError

from src.database import MongoStore, StorageError
import src.feedback_store as feedback
from src.cleaning import MODEL_FEATURES


@pytest.fixture
def store():
    client = MagicMock()
    instance = MongoStore("mongodb://unused/review", client=client)
    collections = {name: MagicMock() for name in ("predictions", "feedback")}
    instance.database.__getitem__.side_effect = collections.__getitem__
    return instance, client.get_default_database.return_value


def test_indexes_are_unique_and_created_once(store):
    instance, db = store
    instance.append("predictions", {"prediction_id": "example"})
    instance.append("predictions", {"prediction_id": "other"})
    for name in ("predictions", "feedback"):
        db[name].create_index.assert_called_once_with(
            "prediction_id", unique=True, name="prediction_id_unique",
        )


def test_numpy_values_are_serializable_and_input_is_not_mutated(store):
    instance, db = store
    row = {"prediction_id": "example", "beds": np.int64(2),
           "correct": np.bool_(True), "latitude": np.float32(41.0), "missing": pd.NA}
    instance.append("predictions", row)
    sent = db["predictions"].insert_one.call_args.args[0]
    assert type(sent["beds"]) is int and type(sent["correct"]) is bool
    assert type(sent["latitude"]) is float and sent["missing"] is None
    assert isinstance(row["beds"], np.int64) and "_id" not in row


def test_reads_keep_column_order_and_hide_mongodb_ids(store):
    instance, db = store
    columns = ["prediction_id", "real_price", "correct"]
    db["feedback"].find.return_value = [{"prediction_id": "example", "real_price": 150., "correct": True}]
    frame = instance.read("feedback", columns)
    assert list(frame) == columns and frame.iloc[0].correct
    projection = db["feedback"].find.call_args.args[1]
    assert projection == {"prediction_id": 1, "real_price": 1, "correct": 1, "_id": 0}
    db["feedback"].find.return_value = []
    assert list(instance.read("feedback", columns)) == columns


def test_optional_columns_may_be_missing_in_older_documents(store):
    instance, db = store
    db["predictions"].find.return_value = [{"prediction_id": "old"},
                                           {"prediction_id": "new", "session_id": "abc"}]
    frame = instance.read("predictions", ["prediction_id", "session_id"], optional=("session_id",))
    assert frame["session_id"].isna().tolist() == [True, False]
    db["predictions"].find.return_value = [{"session_id": "abc"}]
    with pytest.raises(StorageError, match="contrato"):
        instance.read("predictions", ["prediction_id", "session_id"], optional=("session_id",))


def test_incomplete_documents_are_not_silently_filled(store):
    instance, db = store
    db["feedback"].find.return_value = [{"prediction_id": "example"}]
    with pytest.raises(StorageError, match="contrato"):
        instance.read("feedback", ["prediction_id", "real_price"])


def test_duplicate_feedback_is_a_validation_error_not_an_overwrite(store):
    instance, db = store
    db["feedback"].insert_one.side_effect = DuplicateKeyError("internal details")
    with pytest.raises(ValueError, match="Ya se ha guardado"):
        instance.append("feedback", {"prediction_id": "example"})
    db["predictions"].insert_one.side_effect = DuplicateKeyError("internal details")
    with pytest.raises(StorageError):
        instance.append("predictions", {"prediction_id": "example"})


@pytest.mark.parametrize("operation", ["append", "read"])
@pytest.mark.parametrize("driver_error", [OperationFailure, ServerSelectionTimeoutError])
def test_driver_errors_do_not_disclose_connection_or_record_details(store, operation, driver_error):
    instance, db = store
    sensitive = "mongodb://user:secret@private-host/db; example-record"
    if operation == "append":
        db["predictions"].insert_one.side_effect = driver_error(sensitive)
        call = lambda: instance.append("predictions", {"prediction_id": "example"})
    else:
        db["predictions"].find.side_effect = driver_error(sensitive)
        call = lambda: instance.read("predictions", ["prediction_id"])
    with pytest.raises(StorageError) as error:
        call()
    displayed = "".join(traceback.format_exception(error.value))
    assert "secret" not in displayed and "private-host" not in displayed


def test_invalid_configuration_is_sanitized():
    with pytest.raises(StorageError) as error:
        MongoStore("invalid://user:secret@private-host/db")
    assert "secret" not in str(error.value)


def test_backend_selection_and_explicit_csv_directory(tmp_path, monkeypatch):
    assert feedback.storage_backend() == "csv"
    monkeypatch.setenv("MONGODB_URI", "mongodb://unused/example")
    assert feedback.storage_backend() == "mongodb"
    assert feedback.storage_backend(tmp_path) == "csv"
    monkeypatch.setattr(feedback, "mongo_store", lambda *_: pytest.fail("Unexpected MongoDB access"))
    values = {column: 1 for column in MODEL_FEATURES}
    feedback.save_prediction(values, "Medio", {"Económico": .2, "Medio": .6, "Premium": .2},
                             model_name="synthetic", model_version="synthetic",
                             thresholds={"lower": 100., "upper": 200.}, directory=tmp_path)
    assert len(feedback.load_predictions(tmp_path)) == 1


def test_mongodb_error_never_falls_back_to_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://unused/example")
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path))
    unavailable = MagicMock()
    unavailable.append.side_effect = StorageError("MongoDB unavailable")
    unavailable.read.side_effect = StorageError("MongoDB unavailable")
    monkeypatch.setattr(feedback, "mongo_store", lambda *_: unavailable)
    with pytest.raises(StorageError):
        feedback._append(feedback.FEEDBACK, dict.fromkeys(feedback.FEEDBACK_COLUMNS))
    with pytest.raises(StorageError):
        feedback.load_feedback()
    assert not list(tmp_path.iterdir())


def test_mongodb_write_preserves_the_allowlist(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://unused/example")
    backend = MagicMock()
    monkeypatch.setattr(feedback, "mongo_store", lambda *_: backend)
    row = dict.fromkeys(feedback.FEEDBACK_COLUMNS)
    feedback._append(feedback.FEEDBACK, {**row, "unapproved_extra": "not stored"})
    assert backend.append.call_args.args == (feedback.FEEDBACK, row)
