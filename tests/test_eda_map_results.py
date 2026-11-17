"""Validate only the saved aggregate EDA evidence; no raw CSV or GeoJSON needed."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda_map import DISTRICT, ROOM
from src.preprocessing import TARGET_LABELS


ROOT = Path(__file__).resolve().parents[1]


def test_saved_eda_summaries_reconcile_and_preserve_empty_groups():
    folder = ROOT / "docs/results"
    metadata = json.loads((folder / "eda11_metadata.json").read_text())
    assert metadata["analysis_set"] == "train"
    assert metadata["test_inspected"] is metadata["model_trained"] is False
    assert metadata["geometry_joined_to_model"] is False
    assert metadata["model_feature_count"] == 23
    assert metadata["coverage"]["missing_train_neighbourhoods"] == []
    assert metadata["coverage"]["train_neighbourhood_count"] == metadata["coverage"]["matched_train_neighbourhoods"]
    tables = {}
    for name in ["districts", "room_types", "district_room_types"]:
        table = pd.read_csv(folder / f"eda11_{name}.csv")
        tables[name] = table
        assert table.n_listings.sum() == metadata["train_rows"]
        count_columns = [f"count_{label}" for label in TARGET_LABELS]
        assert (table[count_columns] >= 0).all().all()
        np.testing.assert_array_equal(table[count_columns].sum(axis=1), table.n_listings)
        np.testing.assert_array_equal(table.empty_group, table.n_listings.eq(0))
        np.testing.assert_array_equal(table.small_group, table.n_listings.gt(0) & table.n_listings.lt(metadata["small_group_size"]))
        for label in TARGET_LABELS:
            assert table[f"count_{label}"].sum() == metadata["class_counts"][label]
            expected = table[f"count_{label}"] / table.n_listings.replace(0, np.nan) * 100
            np.testing.assert_allclose(table[f"pct_{label}"], expected, equal_nan=True)
    crossed = tables["district_room_types"]
    assert len(crossed) == metadata["crossed_groups"]
    assert crossed.empty_group.sum() == metadata["crossed_empty_groups"]
    assert crossed.small_group.sum() == metadata["crossed_small_groups"]
    for group, name in [(DISTRICT, "districts"), (ROOM, "room_types")]:
        columns = ["n_listings"] + [f"count_{label}" for label in TARGET_LABELS]
        actual = crossed.groupby(group)[columns].sum().sort_index()
        expected = tables[name].set_index(group)[columns].sort_index()
        pd.testing.assert_frame_equal(actual, expected)
    for name, signature in metadata["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == signature
