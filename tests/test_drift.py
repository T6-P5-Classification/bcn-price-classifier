"""Data drift (#36): PSI per variable, alerts and snapshot preparation.

Synthetic data only: no Inside Airbnb download is needed. The saved result of
the real comparison is checked only if it exists.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning import MODEL_FEATURES
from src.drift import (
    CATEGORICAL,
    LARGE,
    MODERATE,
    alerts,
    drift_report,
    key_features,
    level,
    psi,
    snapshot_features,
)

ROOT = Path(__file__).resolve().parents[1]


def listings(n=2000, seed=0, **changes):
    """Synthetic table with the 23 model variables."""
    rng = np.random.default_rng(seed)
    data = {}
    for column in MODEL_FEATURES:
        if column in CATEGORICAL:
            data[column] = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
        elif column.startswith(("has_", "is_")):
            data[column] = rng.integers(0, 2, n)
        else:
            data[column] = rng.normal(10, 2, n)
    frame = pd.DataFrame(data)
    for column, values in changes.items():
        frame[column] = values
    return frame


# --- PSI ------------------------------------------------------------------------

def test_same_distribution_gives_psi_close_to_zero():
    a, b = listings(seed=0), listings(seed=1)  # two samples of the same population
    for column in ["accommodates", "room_type", "has_tv"]:
        assert psi(a[column], b[column]) < MODERATE


def test_shift_of_a_numeric_variable_is_large():
    rng = np.random.default_rng(2)
    assert psi(pd.Series(rng.normal(10, 2, 2000)), pd.Series(rng.normal(14, 2, 2000))) >= LARGE


def test_change_of_category_shares_is_detected():
    reference = pd.Series(["Entire home/apt"] * 650 + ["Private room"] * 350)
    current = pd.Series(["Entire home/apt"] * 350 + ["Private room"] * 650)
    assert psi(reference, current) >= LARGE


def test_categories_never_seen_count_as_drift():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 400 + ["b"] * 400 + ["new type"] * 200)
    assert psi(reference, current) >= MODERATE


def test_missing_values_have_their_own_bin():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0] * 250)
    emptied = np.random.default_rng(5).random(1000) < 0.25
    current = reference.where(~emptied)  # 25 % emptied at random
    assert psi(reference, current) >= LARGE
    assert psi(reference.dropna(), current.dropna()) < MODERATE


def test_psi_is_finite_with_empty_bins_and_needs_data():
    assert np.isfinite(psi(pd.Series([1.0] * 100), pd.Series([5.0] * 100)))
    with pytest.raises(ValueError):
        psi(pd.Series([], dtype=float), pd.Series([1.0]))


def test_full_change_of_an_imbalanced_yes_no_variable_is_large():
    """Review of #80: deciles put 0 and 1 in the same bin and gave PSI 0."""
    reference = pd.Series([1] * 950 + [0] * 50)
    assert psi(reference, pd.Series([0] * 1000)) >= LARGE
    assert psi(reference, pd.Series([1] * 950 + [0] * 50)) == pytest.approx(0)


def test_constant_reference_detects_values_never_seen():
    reference = pd.Series([1.0] * 1000)
    assert psi(reference, pd.Series([1.0] * 1000)) == pytest.approx(0)
    assert psi(reference, pd.Series([1.0] * 500 + [3.0] * 500)) >= LARGE


def test_small_counts_keep_one_bin_per_value():
    reference = pd.Series([1] * 700 + [2] * 200 + [3] * 100)  # e.g. bedrooms
    assert psi(reference, pd.Series([1] * 400 + [2] * 400 + [3] * 200)) >= LARGE
    assert psi(reference, pd.Series([1] * 690 + [2] * 210 + [3] * 100)) < MODERATE


@pytest.mark.parametrize("value, expected",[(0.0, "estable"), (0.0999, "estable"),
                                             (0.10, "moderado"), (0.2499, "moderado"), (0.25, "alto")])
def test_levels_follow_the_usual_thresholds(value, expected):
    assert level(value) == expected


# --- Report and alerts ----------------------------------------------------------

def test_report_has_one_row_per_model_variable_sorted_by_drift():
    reference = listings(seed=0)
    current = listings(seed=1, accommodates=np.random.default_rng(3).normal(15, 2, 2000))
    report = drift_report(reference, current, key=["accommodates"])
    assert sorted(report["feature"]) == sorted(MODEL_FEATURES)
    assert report.iloc[0]["feature"] == "accommodates" and report.iloc[0]["level"] == "alto"
    assert report["psi"].is_monotonic_decreasing


def test_no_alerts_for_the_same_population():
    report = drift_report(listings(seed=0), listings(seed=1), key=["accommodates"])
    assert alerts(report) == []


def test_alerts_explain_whether_values_or_missing_values_changed():
    """Case found with the real June snapshot: same values, different empty fields."""
    reference = listings(seed=0)
    current = listings(seed=1)
    current.loc[current.index[:600], "minimum_nights"] = np.nan      # 30 % empty, values alike
    current["bedrooms"] = np.random.default_rng(4).normal(16, 2, 2000)  # values change
    messages = alerts(drift_report(reference, current, key=["minimum_nights"]))
    by_feature = {m.split(":")[1].split()[0].rstrip(","): m for m in messages}
    assert "vacíos (0% -> 30%)" in by_feature["minimum_nights"]
    assert "variable clave" in by_feature["minimum_nights"]
    assert "vacíos" not in by_feature["bedrooms"]


def test_a_completely_empty_variable_is_not_called_unchanged():
    """Review of #80: with no values left, nothing says whether they changed."""
    reference, current = listings(seed=0), listings(seed=1)
    current["minimum_nights"] = np.nan
    message = next(m for m in alerts(drift_report(reference, current, key=[]))
                   if "minimum_nights" in m)
    assert "valores no comparables" in message
    assert "no los valores" not in message and "nan" not in message


def test_moderate_drift_warns_only_for_key_variables():
    reference = pd.Series(np.repeat(["a", "b"], 500))
    current = pd.Series(np.repeat(["a", "b"], [300, 700]))  # PSI ≈ 0.17
    assert level(psi(reference, current)) == "moderado"
    base = listings(n=1000, seed=0, room_type=reference.to_numpy())
    new = listings(n=1000, seed=1, room_type=current.to_numpy())
    assert alerts(drift_report(base, new, key=[])) == []
    assert alerts(drift_report(base, new, key=["room_type"]))[0].startswith("AVISO: room_type")


def test_report_requires_the_23_variables():
    with pytest.raises(ValueError, match="bedrooms"):
        drift_report(listings(), listings().drop(columns="bedrooms"))


# --- Snapshot preparation -------------------------------------------------------

def test_snapshot_without_a_model_variable_is_rejected(tmp_path):
    """Schema drift: a field that disappears from Inside Airbnb stops the check."""
    raw = pd.DataFrame({"id": [1, 2], "price": ["$100.00", "$200.00"], "amenities": ["[]", "[]"]})
    path = tmp_path / "listings.csv.gz"
    raw.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="model variables"):
        snapshot_features(path)


def test_key_features_come_from_the_champion_importance():
    assert key_features()[:3] == ["minimum_nights", "accommodates", "latitude"]


def test_saved_june_result_if_present():
    path = ROOT / "docs/results/drift_2026-06-24.json"
    if not path.exists():
        pytest.skip("python -m src.drift data/raw/listings_2026-06-24.csv.gz")
    metadata = json.loads(path.read_text())
    assert metadata["reference"]["rows"] == "train"  # the test rows are not used
    report = pd.read_csv(ROOT / "docs/results/drift_2026-06-24.csv")
    assert sorted(report["feature"]) == sorted(MODEL_FEATURES)


def test_app_predictions_can_be_compared_with_train(tmp_path):
    """Contract of docs/feedback.md: stored predictions carry the 23 variables."""
    from src.feedback_store import load_predictions, save_prediction

    reference = listings(n=300, seed=0)
    probs = {"Económico": 0.5, "Medio": 0.3, "Premium": 0.2}
    for row in listings(n=30, seed=1).to_dict("records"):
        save_prediction(row, "Medio", probs, model_name="XGBoost", model_version="v",
                        thresholds={"lower": 122.0, "upper": 250.0}, directory=tmp_path)
    report = drift_report(reference, load_predictions(tmp_path))
    assert sorted(report["feature"]) == sorted(MODEL_FEATURES)
    assert report["psi"].notna().all()
