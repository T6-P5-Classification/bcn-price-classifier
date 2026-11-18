"""Data drift: do new listings still look like the ones the model learned from? (#36)

Reference: the **train** rows of the snapshot the app model was fitted on
(``load_train_test()``; the test rows are left out). Current: another Inside
Airbnb snapshot of Barcelona, prepared with the same cleaning and the same 23
``MODEL_FEATURES``. Prices and predictions are not used: drift looks only at
the inputs, so it can warn before any real price or feedback arrives.

For each variable, the **PSI** (Population Stability Index) compares the share
of listings in each bin:

    PSI = sum over bins of (current% - reference%) * ln(current% / reference%)

- Numeric variables: bins at the reference deciles; with 10 distinct values
  or fewer (yes/no flags, small counts), one bin per value plus one for
  values never seen in the reference.
- Categorical variables: one bin per reference category, plus one bin for
  categories never seen in the reference (e.g. a new property type).
- Missing values always have their own bin.

Usual reading (a rule of thumb, not a statistical test): < 0.10 stable,
0.10-0.25 moderate, >= 0.25 large change -> alert.

Usage:  python -m src.drift data/raw/listings_2026-06-24.csv.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # works with "python src/drift.py" too

from src.cleaning import MODEL_FEATURES, clean_listings, engineer_features  # noqa: E402
from src.preprocessing import RAW_DATA_PATH, load_train_test  # noqa: E402

RESULTS = ROOT / "docs" / "results"
CATEGORICAL = ("neighbourhood_cleansed", "neighbourhood_group_cleansed", "property_type", "room_type")
MODERATE, LARGE = 0.10, 0.25
N_BINS = 10
EPSILON = 1e-4      # share used for an empty bin, so that ln() stays finite
N_KEY_FEATURES = 3  # most important variables of the champion (#16)
NEW, MISSING = "(categoría nueva)", "(vacío)"


# --- Data ---------------------------------------------------------------------

def snapshot_features(path: str | Path) -> pd.DataFrame:
    """The 23 model variables of a snapshot, prepared exactly as for training.

    Raises ``ValueError`` if a model variable cannot be built (schema change).
    """
    featured = engineer_features(clean_listings(pd.read_csv(path, compression="gzip")))
    missing = [c for c in MODEL_FEATURES if c not in featured.columns]
    if missing:
        raise ValueError(f"The snapshot does not provide the model variables: {missing}")
    return featured.loc[:, list(MODEL_FEATURES)].copy()


def reference_features() -> pd.DataFrame:
    """Train rows the app model was fitted on (test rows are discarded)."""
    X_train, _, _, _ = load_train_test()
    return X_train


def key_features(results: Path = RESULTS, n: int = N_KEY_FEATURES) -> list[str]:
    """Most important variables of the champion by permutation importance (#16)."""
    path = results / "champion_feature_importance.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path, comment="#")
    return table.nlargest(n, "importance_mean")["feature"].tolist()


# --- PSI ----------------------------------------------------------------------

def _shares(labels: pd.Series, bins: list) -> np.ndarray:
    counts = labels.value_counts().reindex(bins, fill_value=0).to_numpy(dtype=float)
    if counts.sum() == 0:
        raise ValueError("Cannot compute PSI without data")
    return counts / counts.sum()


def psi_from_shares(reference: np.ndarray, current: np.ndarray) -> float:
    r = np.clip(reference, EPSILON, None)
    c = np.clip(current, EPSILON, None)
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_bins(reference: pd.Series, current: pd.Series, n_bins: int = N_BINS):
    """Bin labels of both samples.

    Few distinct reference values (yes/no flags, constants, small counts): one
    bin per value plus one for values never seen. Quantile edges would merge
    them: with 95 % ones and 5 % zeros every decile is 1, so 0 and 1 would
    share a bin and a full change from 1 to 0 would give PSI 0.
    Otherwise: edges at the reference deciles.
    """
    values = reference.dropna()
    distinct = np.unique(values)
    if len(distinct) <= n_bins:
        known = distinct.tolist()

        def label_values(series: pd.Series) -> pd.Series:
            text = series.astype(object).where(series.isin(known), NEW)
            return text.where(series.notna(), MISSING)

        return label_values(reference), label_values(current), [*known, NEW, MISSING]
    inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = [-np.inf, *inner, np.inf]

    def label(series: pd.Series) -> pd.Series:
        codes = pd.Series(pd.cut(series, edges, right=True, labels=False), index=series.index)
        return codes.astype(object).where(series.notna(), MISSING)

    return label(reference), label(current), [*range(len(edges) - 1), MISSING]


def categorical_bins(reference: pd.Series, current: pd.Series):
    """Bin labels of both samples: reference categories, new ones and missing."""
    known = sorted(reference.dropna().astype(str).unique())

    def label(series: pd.Series) -> pd.Series:
        text = series.astype(str).where(series.notna(), MISSING)
        return text.where(text.isin(known) | (text == MISSING), NEW)

    return label(reference), label(current), [*known, NEW, MISSING]


def psi(reference: pd.Series, current: pd.Series, *, categorical: bool | None = None) -> float:
    """PSI of one variable (0 = same distribution), missing values included."""
    if categorical is None:
        categorical = not pd.api.types.is_numeric_dtype(reference)
    binner = categorical_bins if categorical else numeric_bins
    ref_labels, cur_labels, bins = binner(reference, current)
    return psi_from_shares(_shares(ref_labels, bins), _shares(cur_labels, bins))


def level(value: float) -> str:
    if value >= LARGE:
        return "alto"
    if value >= MODERATE:
        return "moderado"
    return "estable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, key: list[str] = ()) -> pd.DataFrame:
    """One row per model variable, largest drift first."""
    missing = [c for c in MODEL_FEATURES if c not in current.columns]
    if missing:
        raise ValueError(f"Current data lack model variables: {missing}")
    rows = []
    for column in MODEL_FEATURES:
        categorical = column in CATEGORICAL
        ref, cur = reference[column], current[column]
        value = psi(ref, cur, categorical=categorical)
        # Same comparison without missing values: separates a change in the
        # values from a change in how often the field is empty.
        both = ref.notna().any() and cur.notna().any()
        values_psi = psi(ref.dropna(), cur.dropna(), categorical=categorical) if both else float("nan")
        new_share = (
            float((~cur.dropna().astype(str).isin(set(ref.dropna().astype(str)))).mean())
            if categorical and cur.notna().any() else 0.0
        )
        rows.append({
            "feature": column,
            "type": "categórica" if categorical else "numérica",
            "psi": value,
            "psi_without_missing": values_psi,
            "level": level(value),
            "key_feature": column in key,
            "reference_missing": float(ref.isna().mean()),
            "current_missing": float(cur.isna().mean()),
            "current_new_categories": new_share,
            "reference_median": None if categorical else float(ref.median()),
            "current_median": None if categorical else float(cur.median()),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def alerts(report: pd.DataFrame) -> list[str]:
    """Plain-language alerts: large drift anywhere, moderate drift in key variables."""
    messages = []
    for row in report.itertuples():
        if row.level == "estable" or (row.level == "moderado" and not row.key_feature):
            continue
        kind = "ALERTA" if row.level == "alto" else "AVISO"
        key = ", variable clave del modelo," if row.key_feature else ""
        detail = f"PSI {row.psi:.2f}"
        if not np.isfinite(row.psi_without_missing):
            # One side has no values at all: nothing says whether they changed.
            detail += (f"; vacíos {row.reference_missing:.0%} -> {row.current_missing:.0%}, "
                       "valores no comparables")
        elif level(row.psi_without_missing) == "estable" and row.reference_missing != row.current_missing:
            # The values look alike: what changed is how often the field is empty.
            detail += (f"; cambian sobre todo los vacíos ({row.reference_missing:.0%} -> "
                       f"{row.current_missing:.0%}), no los valores (PSI sin vacíos "
                       f"{row.psi_without_missing:.2f})")
        messages.append(f"{kind}: {row.feature}{key} cambia ({detail})")
    return messages


# --- Command line -------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data drift of a snapshot against the model's train data (#36)")
    parser.add_argument("snapshot", type=Path, help="Inside Airbnb listings.csv.gz to compare")
    parser.add_argument("--name", help="Name of the results (default: from the file name)")
    parser.add_argument("--no-save", action="store_true", help="Only print, do not write docs/results")
    args = parser.parse_args(argv)

    reference = reference_features()
    current = snapshot_features(args.snapshot)
    report = drift_report(reference, current, key_features())
    messages = alerts(report)

    shown = report[["feature", "psi", "psi_without_missing", "level", "key_feature",
                    "reference_missing", "current_missing"]].copy()
    shown = shown.round(3)
    print(shown.to_string(index=False))
    print("\n" + ("\n".join(messages) if messages else "Sin alertas: los datos se parecen a los de entrenamiento."))

    if not args.no_save:
        name = args.name or args.snapshot.name.replace("listings_", "").replace(".csv.gz", "")
        report.to_csv(RESULTS / f"drift_{name}.csv", index=False)
        metadata = {
            "issue": 36,
            "reference": {"file": str(RAW_DATA_PATH.relative_to(ROOT)), "rows": "train",
                          "n": len(reference), "raw_sha256": _sha256(RAW_DATA_PATH)},
            "current": {"file": args.snapshot.name, "n": len(current), "raw_sha256": _sha256(args.snapshot)},
            "method": {"metric": "PSI", "numeric_bins": f"{N_BINS} reference quantiles; one bin per value if <= {N_BINS} distinct values",
                       "epsilon": EPSILON, "moderate": MODERATE, "large": LARGE},
            "key_features": key_features(),
            "alerts": messages,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (RESULTS / f"drift_{name}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"\nGuardado en docs/results/drift_{name}.csv y .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
