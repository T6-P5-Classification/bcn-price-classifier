"""Aggregate, train-only class EDA for districts and room types (issue 11).

GeoJSON is a drawing resource, never joined into the model feature table.
No functions here load data, train models or export listing-level records.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.preprocessing import TARGET_LABELS


DISTRICT = "neighbourhood_group_cleansed"
NEIGHBOURHOOD = "neighbourhood_cleansed"
ROOM = "room_type"
SMALL_GROUP_SIZE = 30
MISSING_GROUP = "(Sin dato)"
ROOM_LABELS = {
    "Entire home/apt": "Alojamiento entero",
    "Private room": "Habitación privada",
    "Shared room": "Habitación compartida",
    "Hotel room": "Habitación de hotel",
}
CLASS_COLORS = ("#32856F", "#386EA5", "#8C64A5")


def class_summary(
    X: pd.DataFrame,
    y: pd.Series,
    groups: list[str],
    *,
    small_group_size: int = SMALL_GROUP_SIZE,
) -> pd.DataFrame:
    """Counts and percentages WITHIN each group; empty crossings stay NaN.

    All observed district/room combinations are included in the Cartesian
    product, even if their joint count is zero. Small-group flags are only
    descriptive reminders, not significance tests or privacy guarantees.
    """
    if not groups or len(groups) != len(set(groups)) or not set(groups) <= set(X):
        raise ValueError("Specify distinct grouping columns present in X")
    if small_group_size < 1:
        raise ValueError("small_group_size must be positive")
    if X.empty or not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("Nonempty X and y must have identical, unique indices in order")
    if y.isna().any() or not y.isin(TARGET_LABELS).all():
        raise ValueError("Every target must be one of the agreed real price classes")
    frame = X.loc[:, groups].astype(object).fillna(MISSING_GROUP).copy()
    levels = [sorted(frame[column].unique()) for column in groups]
    if len(groups) == 1:
        index = pd.Index(levels[0], name=groups[0])
    else:
        index = pd.MultiIndex.from_product(levels, names=groups)
    frame["_real_class"] = y.astype(object)
    counts = (
        frame.groupby(groups + ["_real_class"], observed=True).size()
        .unstack("_real_class", fill_value=0)
        .reindex(index=index, columns=list(TARGET_LABELS), fill_value=0)
    )
    totals = counts.sum(axis=1)
    result = pd.DataFrame({"n_listings": totals}, index=index)
    for label in TARGET_LABELS:
        result[f"count_{label}"] = counts[label].astype("int64")
        result[f"pct_{label}"] = counts[label].div(totals.replace(0, np.nan)).mul(100)
    result["small_group"] = totals.gt(0) & totals.lt(small_group_size)
    result["empty_group"] = totals.eq(0)
    return result.reset_index()


def geometry_parts(geometry: dict) -> list:
    """Return polygon rings, supporting both GeoJSON Polygon and MultiPolygon."""
    if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Map requires Polygon or MultiPolygon geometry")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        raise ValueError("Empty polygon coordinates")
    parts = [coordinates] if geometry["type"] == "Polygon" else coordinates
    for polygon in parts:
        if not polygon:
            raise ValueError("Polygon has no exterior ring")
        for ring in polygon:
            points = np.asarray(ring, dtype=float)
            if (points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 2
                    or not np.isfinite(points).all()
                    or not np.array_equal(points[0], points[-1])):
                raise ValueError("Rings must contain finite, closed longitude/latitude pairs")
    return parts


def map_coverage(X: pd.DataFrame, geojson: dict) -> dict:
    """Audit exact neighbourhood-name matching without a spatial/model join.

    Multiple geometry features for one neighbourhood are legitimate pieces if
    their district agrees. They never multiply listing counts.
    """
    if geojson.get("type") != "FeatureCollection" or not geojson.get("features"):
        raise ValueError("Expected a nonempty GeoJSON FeatureCollection")
    if not {NEIGHBOURHOOD, DISTRICT} <= set(X) or X.empty:
        raise ValueError("Training geography columns are required")
    if X[[NEIGHBOURHOOD, DISTRICT]].isna().any().any():
        raise ValueError("Missing training geography: resolve or report before mapping")
    pairs = X[[NEIGHBOURHOOD, DISTRICT]].drop_duplicates()
    if pairs[NEIGHBOURHOOD].duplicated().any():
        raise ValueError("One training neighbourhood maps to multiple districts")
    training = pairs.set_index(NEIGHBOURHOOD)[DISTRICT].to_dict()
    geographic = {}
    occurrences = Counter()
    for feature in geojson["features"]:
        properties = feature.get("properties", {})
        name, district = properties.get("neighbourhood"), properties.get("neighbourhood_group")
        if not isinstance(name, str) or not name or not isinstance(district, str) or not district:
            raise ValueError("GeoJSON needs neighbourhood and neighbourhood_group names")
        geometry_parts(feature.get("geometry"))
        if name in geographic and geographic[name] != district:
            raise ValueError("Geometry pieces disagree about the neighbourhood district")
        geographic[name] = district
        occurrences[name] += 1
    missing = sorted(set(training) - set(geographic))
    if missing:
        raise ValueError(f"Training neighbourhoods without geometry: {missing}")
    mismatched = sorted(name for name in training if training[name] != geographic[name])
    if mismatched:
        raise ValueError(f"Training/GeoJSON district disagreement: {mismatched}")
    return {
        "feature_count": len(geojson["features"]),
        "geo_neighbourhood_count": len(geographic),
        "train_neighbourhood_count": len(training),
        "matched_train_neighbourhoods": len(training),
        "missing_train_neighbourhoods": missing,
        "geo_neighbourhoods_without_train": sorted(set(geographic) - set(training)),
        "multiple_feature_neighbourhoods": {k: v for k, v in occurrences.items() if v > 1},
        "geo_districts_without_train": sorted(set(geographic.values()) - set(training.values())),
    }


def _polygon_path(rings):
    """Respect interior holes using opposite winding, irrespective of source order."""
    from matplotlib.path import Path

    vertices, codes = [], []
    for index, ring in enumerate(rings):
        points = np.asarray(ring, dtype=float)
        signed_area = np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
        if (index == 0 and signed_area < 0) or (index > 0 and signed_area > 0):
            points = points[::-1]
        vertices.extend(points)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(points) - 2) + [Path.CLOSEPOLY])
    return Path(np.asarray(vertices), codes)


def plot_district_maps(geojson: dict, summary: pd.DataFrame):
    """Three district-level class shares on neighbourhood polygons, common 0–100 scale."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch, PathPatch

    values = summary.set_index(DISTRICT)
    pieces = []
    district_points = {}
    for feature in geojson["features"]:
        district = feature["properties"]["neighbourhood_group"]
        for polygon in geometry_parts(feature["geometry"]):
            pieces.append((district, polygon))
            district_points.setdefault(district, []).extend(polygon[0][:-1])
    bounds = np.concatenate([np.asarray(polygon[0]) for _, polygon in pieces])
    positions = {name: np.median(points, axis=0) for name, points in district_points.items()}
    districts = sorted(district_points)
    numbers = {name: index + 1 for index, name in enumerate(districts)}
    norm = colors.Normalize(0, 100)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.2), layout="constrained")
    for ax, label in zip(axes, TARGET_LABELS):
        for district, polygon in pieces:
            share = values.loc[district, f"pct_{label}"] if district in values.index else np.nan
            color = cmap(norm(share)) if np.isfinite(share) else "#D4D4D4"
            ax.add_patch(PathPatch(_polygon_path(polygon), facecolor=color, edgecolor="#555555", linewidth=0.35))
        for district, (longitude, latitude) in positions.items():
            ax.text(longitude, latitude, str(numbers[district]), ha="center", va="center", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#666666", "boxstyle": "round,pad=0.18"})
        ax.set(xlim=(bounds[:, 0].min() - 0.01, bounds[:, 0].max() + 0.01),
               ylim=(bounds[:, 1].min() - 0.008, bounds[:, 1].max() + 0.008),
               title=f"{label} · categoría REAL", xlabel="Longitud (grados)")
        ax.set_aspect(1 / np.cos(np.deg2rad(bounds[:, 1].mean())))
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Latitud (grados)")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), shrink=0.65,
                 label="Porcentaje de anuncios del distrito en esa categoría REAL")
    fig.suptitle("Composición de precio por distrito · solo anuncios de entrenamiento", fontsize=15)
    handles = [Patch(facecolor="white", edgecolor="#777777", label=f"{numbers[name]}. {name}") for name in districts]
    handles.append(Patch(facecolor="#D4D4D4", label="Distrito sin observaciones"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10, frameon=False)
    return fig


def plot_group_bars(summary: pd.DataFrame, group: str, *, title: str):
    """Stacked shares with support shown in labels; denominators are whole groups."""
    import matplotlib.pyplot as plt

    table = summary.set_index(group)
    proportions = table[[f"pct_{label}" for label in TARGET_LABELS]].copy()
    proportions.columns = list(TARGET_LABELS)
    proportions.index = [f"{ROOM_LABELS.get(name, name)} (n={int(n):,})" for name, n in zip(table.index, table.n_listings)]
    fig, ax = plt.subplots(figsize=(11, max(4.2, len(table) * 0.43 + 1.8)), layout="constrained")
    proportions.plot.barh(stacked=True, color=CLASS_COLORS, ax=ax, width=0.75)
    ax.set(xlim=(0, 100), xlabel="Porcentaje dentro del grupo indicado (cada barra suma 100 %)", ylabel="Grupo y número de anuncios de entrenamiento", title=title)
    ax.invert_yaxis()
    ax.legend(title="Categoría REAL de precio", bbox_to_anchor=(1.01, 1), loc="upper left")
    return fig


def plot_crossed_shares(summary: pd.DataFrame, *, small_group_size: int = SMALL_GROUP_SIZE):
    """Show class composition within district × room groups, with empty/rare cells explicit."""
    import matplotlib.pyplot as plt

    districts = sorted(summary[DISTRICT].unique())
    rooms = sorted(summary[ROOM].unique())
    table = summary.set_index([DISTRICT, ROOM])
    index = pd.MultiIndex.from_product([districts, rooms], names=[DISTRICT, ROOM])
    table = table.reindex(index)
    n = table.n_listings.to_numpy().reshape(len(districts), len(rooms))
    cmap = plt.get_cmap("viridis").with_extremes(bad="#D4D4D4")
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.2), sharey=True, layout="constrained")
    for ax, label in zip(axes, TARGET_LABELS):
        percentages = table[f"pct_{label}"].to_numpy().reshape(len(districts), len(rooms))
        image = ax.imshow(np.ma.masked_invalid(percentages), vmin=0, vmax=100, cmap=cmap, aspect="auto")
        for row in range(len(districts)):
            for col in range(len(rooms)):
                count = int(n[row, col])
                if count == 0:
                    text = "Sin datos\nn=0"
                else:
                    marker = "†" if count < small_group_size else ""
                    text = f"{percentages[row, col]:.0f} %\nn={count}{marker}"
                color = "white" if np.isfinite(percentages[row, col]) and percentages[row, col] < 65 else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(rooms)), [ROOM_LABELS.get(room, room).replace(" ", "\n", 1) for room in rooms], fontsize=9)
        ax.set_yticks(range(len(districts)), districts, fontsize=9)
        ax.set(title=f"{label} · categoría REAL", xlabel="Modalidad alquilada (room_type)")
    axes[0].set_ylabel("Distrito")
    fig.colorbar(image, ax=axes.tolist(), shrink=0.65, label="Porcentaje dentro de cada distrito × modalidad")
    fig.suptitle(f"Solo entrenamiento · † n < {small_group_size}: grupo pequeño, no prueba estadística · gris: sin datos", fontsize=13)
    return fig
