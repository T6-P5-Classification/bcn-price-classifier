"""Issue 11 checks with synthetic groups and polygons; no Airbnb files required."""

import copy
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.eda_map import (
    DISTRICT, NEIGHBOURHOOD, ROOM, MISSING_GROUP,
    _polygon_path, class_summary, geometry_parts, map_coverage,
    plot_crossed_shares, plot_district_maps, plot_group_bars,
)
from src.preprocessing import TARGET_LABELS


@pytest.fixture
def training():
    index = [11, 12, 21, 22, 31, 32]
    X = pd.DataFrame({
        NEIGHBOURHOOD: ["n1", "n1", "n2", "n2", "n3", "n3"],
        DISTRICT: ["A", "A", "A", "A", "B", "B"],
        ROOM: ["Entire home/apt", "Entire home/apt", "Private room", "Private room", "Entire home/apt", "Entire home/apt"],
    }, index=index)
    y = pd.Series(["Económico", "Económico", "Medio", "Premium", "Premium", "Medio"], index=index)
    return X, y


def feature(name, district, offset=0):
    ring = [[offset, 0], [offset + 1, 0], [offset + 1, 1], [offset, 1], [offset, 0]]
    return {"type": "Feature", "properties": {"neighbourhood": name, "neighbourhood_group": district},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


@pytest.fixture
def geometry():
    # n1 has two geometry objects; n4 is a public district without train rows.
    return {"type": "FeatureCollection", "features": [
        feature("n1", "A"), feature("n1", "A", 1), feature("n2", "A", 2),
        feature("n3", "B", 3), feature("n4", "C", 4),
    ]}


def test_percentages_have_group_not_class_denominator(training):
    X, y = training
    result = class_summary(X, y, [DISTRICT], small_group_size=3).set_index(DISTRICT)
    assert result.loc["A", "n_listings"] == 4
    assert result.loc["A", "count_Económico"] == 2
    assert result.loc["A", "pct_Económico"] == 50  # not 100% of all Económico
    assert result.loc["A", "pct_Medio"] == result.loc["A", "pct_Premium"] == 25
    assert result.loc["B", "count_Económico"] == result.loc["B", "pct_Económico"] == 0
    assert bool(result.loc["B", "small_group"])
    assert not bool(result.loc["A", "small_group"])
    assert result.n_listings.sum() == len(y)
    np.testing.assert_allclose(result[[f"pct_{c}" for c in TARGET_LABELS]].sum(axis=1), 100)


def test_crossed_empty_groups_are_explicit_not_zero_percent(training):
    X, y = training
    result = class_summary(X, y, [DISTRICT, ROOM], small_group_size=2)
    assert len(result) == 4
    empty = result[(result[DISTRICT] == "B") & (result[ROOM] == "Private room")].iloc[0]
    assert empty.n_listings == 0 and empty.empty_group
    assert not empty.small_group
    assert all(empty[f"count_{c}"] == 0 for c in TARGET_LABELS)
    assert all(pd.isna(empty[f"pct_{c}"]) for c in TARGET_LABELS)
    # n == threshold is not flagged; no group is silently filtered.
    assert not result.small_group.any()
    assert result.n_listings.sum() == len(y)


def test_missing_group_kept_and_missing_real_classes_rejected(training):
    X, y = training
    X = X.copy()
    X.loc[11, ROOM] = None
    result = class_summary(X, y, [ROOM])
    assert MISSING_GROUP in result[ROOM].tolist()
    assert result.n_listings.sum() == len(y)
    bad = y.copy()
    bad.loc[11] = None
    with pytest.raises(ValueError, match="Every target"):
        class_summary(X, bad, [ROOM])
    bad.loc[11] = "Unknown"
    with pytest.raises(ValueError, match="Every target"):
        class_summary(X, bad, [ROOM])


def test_summary_rejects_index_mismatch_duplicates_and_invalid_inputs(training):
    X, y = training
    with pytest.raises(ValueError, match="unique indices"):
        class_summary(X, y.iloc[::-1], [DISTRICT])
    duplicate_X, duplicate_y = X.copy(), y.copy()
    duplicate_X.index = duplicate_y.index = [1, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="unique indices"):
        class_summary(duplicate_X, duplicate_y, [DISTRICT])
    for groups in [[], [DISTRICT, DISTRICT], ["not_a_column"]]:
        with pytest.raises(ValueError, match="grouping columns"):
            class_summary(X, y, groups)
    with pytest.raises(ValueError, match="positive"):
        class_summary(X, y, [ROOM], small_group_size=0)


def test_aggregation_and_coverage_do_not_modify_features(training, geometry):
    X, y = training
    original_X, original_y, original_geo = X.copy(deep=True), y.copy(deep=True), copy.deepcopy(geometry)
    class_summary(X, y, [DISTRICT, ROOM])
    report = map_coverage(X, geometry)
    assert report["feature_count"] == 5
    assert report["geo_neighbourhood_count"] == 4
    assert report["train_neighbourhood_count"] == report["matched_train_neighbourhoods"] == 3
    assert report["multiple_feature_neighbourhoods"] == {"n1": 2}
    assert report["geo_neighbourhoods_without_train"] == ["n4"]
    assert report["geo_districts_without_train"] == ["C"]
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
    assert geometry == original_geo


def test_missing_geometry_is_rejected_not_silently_dropped(training, geometry):
    X, _ = training
    geometry["features"] = [f for f in geometry["features"] if f["properties"]["neighbourhood"] != "n3"]
    with pytest.raises(ValueError, match="without geometry"):
        map_coverage(X, geometry)


def test_conflicting_districts_are_rejected(training, geometry):
    X, _ = training
    geometry["features"][0]["properties"]["neighbourhood_group"] = "Z"
    with pytest.raises(ValueError, match="pieces disagree"):
        map_coverage(X, geometry)
    geometry["features"][1]["properties"]["neighbourhood_group"] = "Z"
    with pytest.raises(ValueError, match="district disagreement"):
        map_coverage(X, geometry)
    bad = X.copy()
    bad.loc[12, DISTRICT] = "Z"
    with pytest.raises(ValueError, match="multiple districts"):
        map_coverage(bad, geometry)


def test_missing_training_geography_requires_explicit_handling(training, geometry):
    X, _ = training
    bad = X.copy()
    bad.loc[11, NEIGHBOURHOOD] = None
    with pytest.raises(ValueError, match="Missing training geography"):
        map_coverage(bad, geometry)


def test_polygon_and_multipolygon_and_holes():
    exterior = [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]  # clockwise
    hole = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]  # counterclockwise
    polygon = {"type": "Polygon", "coordinates": [exterior, hole]}
    multipolygon = {"type": "MultiPolygon", "coordinates": [[exterior, hole], [exterior]]}
    assert len(geometry_parts(polygon)) == 1
    assert len(geometry_parts(multipolygon)) == 2
    path = _polygon_path(geometry_parts(polygon)[0])
    parts = [path.vertices[:5], path.vertices[5:]]
    areas = [np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]) for p in parts]
    assert areas[0] > 0 and areas[1] < 0


@pytest.mark.parametrize("bad", [
    None,
    {"type": "Point", "coordinates": [0, 0]},
    {"type": "Polygon", "coordinates": []},
    {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]},
    {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, np.nan], [0, 0]]]},
])
def test_bad_geometry_is_rejected(bad):
    with pytest.raises(ValueError):
        geometry_parts(bad)


def test_figures_render_with_common_scale_and_empty_cells(training, geometry):
    X, y = training
    districts = class_summary(X, y, [DISTRICT])
    crossed = class_summary(X, y, [DISTRICT, ROOM])
    map_figure = plot_district_maps(geometry, districts)
    bar_figure = plot_group_bars(districts, DISTRICT, title="Synthetic real-class groups")
    cross_figure = plot_crossed_shares(crossed)
    assert len(map_figure.axes) == len(cross_figure.axes) == 4  # 3 panels + colorbar
    assert len(bar_figure.axes) == 1
    for ax in cross_figure.axes[:3]:
        assert ax.images[0].get_clim() == (0, 100)
        assert any("Sin datos" in text.get_text() for text in ax.texts)
    assert bar_figure.axes[0].get_xlim() == (0, 100)
    for fig in [map_figure, bar_figure, cross_figure]:
        fig.canvas.draw()
        plt.close(fig)


def test_notebook_loading_never_inspects_reserved_objects(monkeypatch):
    import src.preprocessing as preprocessing
    from src.cleaning import MODEL_FEATURES

    class DoNotInspect:
        def __getattribute__(self, name):
            raise AssertionError(f"Reserved object was inspected: {name}")

    X = pd.DataFrame(np.zeros((6, 23)), columns=MODEL_FEATURES)
    y = pd.Series(list(TARGET_LABELS) * 2)
    monkeypatch.setattr(preprocessing, "load_train_test", lambda **_: (
        X, DoNotInspect(), y, DoNotInspect(), {"lower": 1.0, "upper": 2.0}
    ))
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", lambda _: None)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/03_eda_map.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            assert not any(isinstance(node, ast.Attribute) and node.attr in {"fit", "fit_transform", "predict", "predict_proba"} for node in ast.walk(tree))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "load-train")["source"])
    namespace = {}
    exec(compile(source, "eda11/load-train", "exec"), namespace)
    assert "reserved_X" not in namespace and "reserved_y" not in namespace
    assert namespace["X_train"] is X
