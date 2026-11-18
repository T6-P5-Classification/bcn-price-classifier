"""A/B test champion vs challenger (#35): assignment, challenger model and analysis.

The analysis is tested with synthetic feedback: the app has no real users yet.
The app itself (assignment per session, recording) is tested in test_app.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ab_test import (
    CHALLENGER,
    CHAMPION,
    MIN_FEEDBACK_PER_MODEL,
    assign_variant,
    challenger_share,
    compare_variants,
    experiment,
    summary,
    variant_artifact,
)
from src.train_app_model import challenger_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"
ARTIFACT = {"model": "xgb", "model_name": "XGBoost", "thresholds": {"lower": 1, "upper": 2},
            "challenger": {"model": "lgbm", "model_name": "LightGBM", "test_metrics": None}}


# --- Assignment -------------------------------------------------------------

def test_share_defaults_to_half_and_reads_ab_split():
    assert challenger_share(ARTIFACT, {}) == 0.5
    assert challenger_share(ARTIFACT, {"AB_SPLIT": "0.2"}) == 0.2
    assert challenger_share({"model": "xgb"}, {"AB_SPLIT": "1"}) == 0.0  # no challenger


@pytest.mark.parametrize("value", ["1.5", "-0.1", "mitad", "nan"])
def test_invalid_split_is_rejected(value):
    with pytest.raises(ValueError, match="AB_SPLIT"):
        challenger_share(ARTIFACT, {"AB_SPLIT": value})


def test_assignment_follows_the_share():
    assert assign_variant(0.49, 0.5) == CHALLENGER
    assert assign_variant(0.5, 0.5) == CHAMPION
    assert assign_variant(0.0, 0.0) == CHAMPION     # 0 % challenger
    assert assign_variant(0.999, 1.0) == CHALLENGER  # 100 % challenger


def test_random_assignment_splits_users_about_in_half():
    rng = np.random.default_rng(0)
    variants = [assign_variant(rng.random(), 0.5) for _ in range(10_000)]
    assert variants.count(CHALLENGER) / len(variants) == pytest.approx(0.5, abs=0.02)


def test_variant_artifact_swaps_only_the_model_and_its_description():
    assert variant_artifact(ARTIFACT, CHAMPION) is ARTIFACT
    challenger = variant_artifact(ARTIFACT, CHALLENGER)
    assert challenger["model"] == "lgbm" and challenger["model_name"] == "LightGBM"
    assert challenger["thresholds"] == ARTIFACT["thresholds"]  # same classes and form
    assert ARTIFACT["model"] == "xgb"  # the original artifact is not modified
    with pytest.raises(ValueError):
        variant_artifact(ARTIFACT, "C")


# --- Challenger model ---------------------------------------------------------

def test_challenger_uses_the_selected_lightgbm_parameters():
    details = json.loads((RESULTS / "lightgbm_details.json").read_text())
    params = challenger_model().named_steps["model"].get_params()
    for key, value in {**details["fixed_params"], **details["selected_params"]}.items():
        assert params[key] == pytest.approx(value)


def test_challenger_must_come_from_the_same_data_and_parameters(tmp_path):
    with pytest.raises(ValueError, match="different data"):
        challenger_model(raw_sha256="b" * 64)
    details = json.loads((RESULTS / "lightgbm_details.json").read_text())
    details["selected_params"]["num_leaves"] = 8
    (tmp_path / "lightgbm_details.json").write_text(json.dumps(details))
    (tmp_path / "lightgbm_comparison.csv").write_text((RESULTS / "lightgbm_comparison.csv").read_text())
    with pytest.raises(ValueError, match="differ"):
        challenger_model(tmp_path)


# --- Analysis -----------------------------------------------------------------

MODELS = {CHAMPION: ("XGBoost", "xgb-v2"), CHALLENGER: ("LightGBM", "lgbm-v2")}


def feedback(champion_hits, champion_n, challenger_hits, challenger_n, *,
             versions=("xgb-v2", "lgbm-v2"), prefix="s", repeats=1):
    """Synthetic labelled answers; by default one answer per session."""
    rows = []
    for variant, name, version, hits, n in [
        ("champion", "XGBoost", versions[0], champion_hits, champion_n),
        ("challenger", "LightGBM", versions[1], challenger_hits, challenger_n),
    ]:
        for i in range(n):
            rows.append({"prediction_id": f"{prefix}-{variant}-{i}", "created_at": f"2026-11-18T10:{i // 60 % 60:02d}:{i % 60:02d}",
                         "variant": variant, "model_name": name, "model_version": version,
                         "session_id": f"{prefix}-{variant}-{i // repeats}", "correct": i < hits})
    return pd.DataFrame(rows, columns=["prediction_id", "created_at", "variant", "model_name",
                                       "model_version", "session_id", "correct"])


def test_experiment_comes_from_the_artifact():
    artifact = {"model_name": "XGBoost", "model_version": "xgb-v2",
                "challenger": {"model_name": "LightGBM", "model_version": "lgbm-v2"}}
    assert experiment(artifact) == MODELS
    with pytest.raises(ValueError, match="no challenger"):
        experiment({"model_name": "XGBoost", "model_version": "xgb-v2"})


def test_answers_of_other_models_or_versions_are_left_out():
    """Review of #79: old wrong answers of the provisional model made the challenger win."""
    current = feedback(100, 100, 100, 100)                       # a tie: 100/100 each
    old = feedback(0, 300, 0, 0, versions=("provisional", "-"), prefix="old")
    old["model_name"] = "Regresión logística"
    result = compare_variants(pd.concat([current, old], ignore_index=True), MODELS)
    assert result["decision"] == "no_difference"
    assert result["champion"]["n"] == 100 and result["excluded"]["other_models"] == 300


def test_repeated_answers_of_one_session_count_once():
    """Review of #79: two sessions with 100 repetitions each are two sessions."""
    data = feedback(100, 100, 0, 100, repeats=100)
    result = compare_variants(data, MODELS)
    assert (result["champion"]["n"], result["challenger"]["n"]) == (1, 1)
    assert result["excluded"]["repeated"] == 198
    assert result["decision"] == "insufficient_data"


def test_answers_without_session_are_left_out():
    data = feedback(100, 100, 100, 100)
    data.loc[data.index[:10], "session_id"] = None
    result = compare_variants(data, MODELS)
    assert result["excluded"]["no_session"] == 10 and result["champion"]["n"] == 90


def test_few_answers_never_declare_a_winner():
    """Review question in class: 5/5 against 3/5 is not evidence."""
    result = compare_variants(feedback(5, 5, 3, 5), MODELS)
    assert result["decision"] == "insufficient_data"
    assert result["champion"]["accuracy"] == 1.0 and result["challenger"]["accuracy"] == 0.6
    assert result["interval_95"] is None


def test_small_difference_with_enough_answers_is_no_difference():
    result = compare_variants(feedback(78, 100, 72, 100), MODELS)
    assert result["decision"] == "no_difference"
    low, high = result["interval_95"]
    assert low < 0 < high and result["difference"] == pytest.approx(-0.06)


@pytest.mark.parametrize("data, decision", [
    (feedback(700, 1000, 800, 1000), "challenger_better"),
    (feedback(800, 1000, 700, 1000), "champion_better"),
])
def test_clear_differences_with_many_answers_are_detected(data, decision):
    assert compare_variants(data, MODELS)["decision"] == decision


def test_all_correct_in_both_groups_is_not_a_difference():
    """100 % vs 100 %: the adjusted interval is not zero-width."""
    result = compare_variants(feedback(150, 150, 150, 150), MODELS)
    assert result["decision"] == "no_difference"
    low, high = result["interval_95"]
    assert low < 0 < high


def test_minimum_is_per_model():
    data = feedback(900, 1000, 90, MIN_FEEDBACK_PER_MODEL - 1)
    assert compare_variants(data, MODELS)["decision"] == "insufficient_data"


def test_summary_is_readable():
    text = summary(compare_variants(feedback(78, 100, 72, 100), MODELS))
    assert "XGBoost (champion): 78/100" in text and "LightGBM (challenger): 72/100" in text
    assert "IC 95 %" in text and "se mantiene el champion" in text
    assert "Fuera del análisis" not in text
    assert "Faltan respuestas" in summary(compare_variants(feedback(0, 0, 0, 0), MODELS))
    repeated = summary(compare_variants(feedback(1, 2, 1, 2, repeats=2), MODELS))
    assert "2 repetidas en una sesión" in repeated


def test_analysis_reads_the_stored_feedback(tmp_path):
    """End to end with the real storage: predictions + feedback -> comparison."""
    from src.cleaning import MODEL_FEATURES
    from src.feedback_store import load_labelled, save_feedback, save_prediction

    values = {c: 1 for c in MODEL_FEATURES}
    values.update({"neighbourhood_cleansed": "el Raval", "neighbourhood_group_cleansed": "Ciutat Vella",
                   "property_type": "Entire rental unit", "room_type": "Entire home/apt",
                   "latitude": 41.38, "longitude": 2.17})
    thresholds = {"lower": 122.0, "upper": 250.0}
    probs = {"Económico": 0.6, "Medio": 0.3, "Premium": 0.1}
    models = {CHAMPION: ("XGBoost", "XGBoost-v"), CHALLENGER: ("LightGBM", "LightGBM-v")}
    for variant, name, price, session in [("champion", "XGBoost", 100.0, "a"),
                                          ("champion", "XGBoost", 300.0, "b"),
                                          ("champion", "XGBoost", 300.0, "b"),  # repeated
                                          ("challenger", "LightGBM", 90.0, "c")]:
        pid = save_prediction(values, "Económico", probs, model_name=name, model_version=f"{name}-v",
                              thresholds=thresholds, variant=variant, session_id=session,
                              directory=tmp_path)
        save_feedback(pid, price, directory=tmp_path)
    result = compare_variants(load_labelled(tmp_path), models, min_per_model=1)
    assert (result["champion"]["n"], result["champion"]["correct"]) == (2, 1)
    assert (result["challenger"]["n"], result["challenger"]["correct"]) == (1, 1)
    assert result["excluded"]["repeated"] == 1


def test_new_sessions_count_after_an_old_csv_without_session_id(tmp_path):
    """Review of #79: 100 new sessions per model stored after an old CSV file
    must be counted, not dropped for lack of session_id."""
    from src.cleaning import MODEL_FEATURES
    from src.feedback_store import load_labelled, save_feedback, save_prediction

    values = {c: 1 for c in MODEL_FEATURES}
    values.update({"neighbourhood_cleansed": "el Raval", "neighbourhood_group_cleansed": "Ciutat Vella",
                   "property_type": "Entire rental unit", "room_type": "Entire home/apt",
                   "latitude": 41.38, "longitude": 2.17})
    probs = {"Económico": 0.6, "Medio": 0.3, "Premium": 0.1}
    thresholds = {"lower": 122.0, "upper": 250.0}

    def answer(variant, name, session):
        pid = save_prediction(values, "Económico", probs, model_name=name, model_version=f"{name}-v",
                              thresholds=thresholds, variant=variant, session_id=session,
                              directory=tmp_path)
        save_feedback(pid, 100.0, directory=tmp_path)

    answer("champion", "XGBoost", None)  # historical row, then the file loses the column
    path = tmp_path / "predictions.csv"
    pd.read_csv(path).drop(columns="session_id").to_csv(path, index=False)
    for i in range(100):
        answer("champion", "XGBoost", f"x{i}")
        answer("challenger", "LightGBM", f"l{i}")
    models = {CHAMPION: ("XGBoost", "XGBoost-v"), CHALLENGER: ("LightGBM", "LightGBM-v")}
    result = compare_variants(load_labelled(tmp_path), models)
    assert (result["champion"]["n"], result["challenger"]["n"]) == (100, 100)
    assert result["excluded"]["no_session"] == 1
