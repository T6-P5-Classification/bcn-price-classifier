"""A/B test of the app: champion vs challenger (#35).

How it works:

1. **Random assignment.** When a user opens the app, a coin toss decides
   which model answers: the champion (XGBoost, chosen in #22) or the
   challenger (LightGBM, a practical tie in cross-validation). The share of
   users that get the challenger is ``AB_SPLIT`` (default 0.5).
2. **Sticky.** The toss happens once per session, so the same user always
   gets the same model (the app keeps it in ``st.session_state``).
3. **Recorded.** Every prediction is stored with its ``variant`` (and
   ``model_name``), so each piece of feedback can be credited to the model
   that made the prediction (contract in docs/feedback.md).
4. **Compared with care.** ``compare_variants`` compares the share of correct
   predictions of each model with a 95 % interval for the difference. It only
   declares a winner when there is enough feedback in both groups and the
   interval excludes zero; otherwise the decision is to keep the champion.

The test set is not used: the challenger is fitted on train only, with the
hyperparameters selected in its own cross-validation (#21).
"""

from __future__ import annotations

import math
import os

import pandas as pd

CHAMPION, CHALLENGER = "champion", "challenger"
DEFAULT_SPLIT = 0.5
MIN_FEEDBACK_PER_MODEL = 100   # below this, a difference is mostly noise (±12 points)
Z_95 = 1.96


def challenger_share(artifact: dict, environ=os.environ) -> float:
    """Share of sessions that get the challenger (0 if the artifact has none)."""
    if "challenger" not in artifact:
        return 0.0
    raw = environ.get("AB_SPLIT", DEFAULT_SPLIT)
    try:
        share = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"AB_SPLIT must be a number between 0 and 1, got {raw!r}") from None
    if not 0 <= share <= 1:
        raise ValueError(f"AB_SPLIT must be between 0 and 1, got {share}")
    return share


def assign_variant(coin: float, share: float) -> str:
    """Variant for a session, from a uniform random number in [0, 1)."""
    return CHALLENGER if coin < share else CHAMPION


def variant_artifact(artifact: dict, variant: str) -> dict:
    """Artifact as seen by the app for one variant.

    The challenger shares form options and thresholds with the champion; only
    the model and what describes it (name, version, labels, metrics) change.
    """
    if variant == CHAMPION:
        return artifact
    if variant != CHALLENGER:
        raise ValueError(f"Unknown variant: {variant!r}")
    return {**artifact, **artifact["challenger"]}


# --- Analysis of the feedback -------------------------------------------------

def _group(labelled: pd.DataFrame, variant: str) -> dict:
    rows = labelled.loc[labelled["variant"] == variant]
    n, hits = len(rows), int(rows["correct"].astype(bool).sum())
    names = sorted(rows["model_name"].unique()) if "model_name" in rows else []
    return {"model": ", ".join(names) or variant, "n": n, "correct": hits,
            "accuracy": hits / n if n else None}


def compare_variants(
    labelled: pd.DataFrame, *, min_per_model: int = MIN_FEEDBACK_PER_MODEL
) -> dict:
    """Compare the share of correct predictions of champion and challenger.

    ``labelled`` is ``feedback_store.load_labelled()``: one row per prediction
    with a real price, with ``variant``, ``model_name`` and ``correct``.

    The 95 % interval of the difference (challenger - champion) uses the
    Agresti-Caffo adjustment (one extra hit and one extra miss per group),
    which behaves better than the plain formula with few data or with 0 % /
    100 % accuracy. Decisions:

    - ``"insufficient_data"``: fewer than ``min_per_model`` answers in a group.
    - ``"challenger_better"`` / ``"champion_better"``: the interval excludes 0.
    - ``"no_difference"``: the interval includes 0; keep the champion.
    """
    a, b = _group(labelled, CHAMPION), _group(labelled, CHALLENGER)
    result = {"champion": a, "challenger": b, "difference": None, "interval_95": None}
    if min(a["n"], b["n"]) < min_per_model:
        return {**result, "decision": "insufficient_data"}

    pa = (a["correct"] + 1) / (a["n"] + 2)
    pb = (b["correct"] + 1) / (b["n"] + 2)
    se = math.sqrt(pa * (1 - pa) / (a["n"] + 2) + pb * (1 - pb) / (b["n"] + 2))
    difference = b["accuracy"] - a["accuracy"]
    low, high = (pb - pa) - Z_95 * se, (pb - pa) + Z_95 * se
    if low > 0:
        decision = "challenger_better"
    elif high < 0:
        decision = "champion_better"
    else:
        decision = "no_difference"
    return {**result, "difference": difference, "interval_95": (low, high), "decision": decision}


DECISIONS_ES = {
    "insufficient_data": "Faltan respuestas: se mantiene el champion.",
    "no_difference": "Sin diferencia clara entre los modelos: se mantiene el champion.",
    "champion_better": "El champion acierta más: se mantiene.",
    "challenger_better": "El challenger acierta más: candidato a sustituir al champion "
                         "(decisión del equipo, tras revisar los mínimos de calidad).",
}


def summary(result: dict) -> str:
    """Short Spanish summary of ``compare_variants`` for the team."""
    lines = []
    for role in ("champion", "challenger"):
        g = result[role]
        accuracy = "—" if g["accuracy"] is None else f"{g['accuracy']:.1%}"
        lines.append(f"{g['model']} ({role}): {g['correct']}/{g['n']} aciertos ({accuracy})")
    if result["interval_95"] is not None:
        low, high = result["interval_95"]
        lines.append(f"Diferencia (challenger - champion): {result['difference']:+.1%}, "
                     f"IC 95 % [{low:+.1%}, {high:+.1%}]")
    lines.append(DECISIONS_ES[result["decision"]])
    return "\n".join(lines)


if __name__ == "__main__":  # python -m src.ab_test
    from src.feedback_store import load_labelled

    print(summary(compare_variants(load_labelled())))
