"""MongoDB backend for the shared prediction/feedback interface (#27).

No connection is opened on import. Never log connection strings or driver
errors: they can contain credentials, server addresses or user records.
"""

from functools import lru_cache
from threading import Lock

import numpy as np
import pandas as pd
from bson.errors import BSONError
from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError


TABLES = ("predictions", "feedback")
TIMEOUT_MS = 3000


class StorageError(Exception):
    """Storage unavailable or inconsistent; safe to catch in the app."""


def _scalar(value):
    """Keep the flat contract, converting NumPy scalars to BSON-compatible types."""
    if value is pd.NA:
        return None
    return value.item() if isinstance(value, np.generic) else value


class MongoStore:
    """One reusable client; uniqueness is enforced by MongoDB, not a local lock."""

    def __init__(self, uri, *, client=None):
        try:
            self.client = client if client is not None else MongoClient(
                uri, connect=False, serverSelectionTimeoutMS=TIMEOUT_MS,
                connectTimeoutMS=TIMEOUT_MS, socketTimeoutMS=TIMEOUT_MS,
            )
            self.database = self.client.get_default_database("bcn_price_classifier")
        except (PyMongoError, ValueError, TypeError):
            raise StorageError("La configuración de MongoDB no es válida.") from None
        self._index_lock = Lock()
        self._indexes_ready = False

    def _collection(self, table):
        if table not in TABLES:
            raise ValueError("Unknown storage table")
        with self._index_lock:
            if not self._indexes_ready:
                for name in TABLES:
                    self.database[name].create_index(
                        "prediction_id", unique=True, name="prediction_id_unique",
                    )
                self._indexes_ready = True
        return self.database[table]

    def append(self, table, row):
        try:
            document = {key: _scalar(value) for key, value in row.items()}
            self._collection(table).insert_one(document)
        except DuplicateKeyError:
            if table == "feedback":
                raise ValueError("Ya se ha guardado una respuesta para esta predicción.") from None
            raise StorageError("La predicción ya existe en MongoDB.") from None
        except (PyMongoError, BSONError, TypeError, OverflowError):
            raise StorageError("No se ha podido guardar en MongoDB. Inténtalo más tarde.") from None

    def read(self, table, columns):
        try:
            projection = {column: 1 for column in columns}
            projection["_id"] = 0
            rows = list(self._collection(table).find(
                {}, projection, sort=[("_id", ASCENDING)],
            ))
        except (PyMongoError, BSONError):
            raise StorageError("No se han podido leer los datos de MongoDB.") from None
        if any(set(row) != set(columns) for row in rows):
            raise StorageError("Los registros de MongoDB no cumplen el contrato de datos.")
        return pd.DataFrame(rows, columns=columns)

    def close(self):
        self.client.close()


@lru_cache(maxsize=1)
def mongo_store(uri):
    """Reuse the client/pool for the configured URI within this app process."""
    return MongoStore(uri)
