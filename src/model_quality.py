"""Minimum quality a classification model must keep (#32).

These thresholds are a regression guard, not a selection criterion: the
champion was chosen in #22 with the agreed rule (best mean validation macro F1
with gap < 0.05). They were set after seeing the results, a little below them,
so that a clearly worse model (for example after retraining in #34) is
detected while small random differences are not.

``quality_problems`` can be reused by #34 before replacing the app model.
"""

from __future__ import annotations

import math

from src.preprocessing import TARGET_LABELS

MIN_F1_MACRO = 0.70             # champion: 0.763 CV, 0.765 test; logistic baseline 0.729
MIN_CLASS_F1 = 0.60             # weakest class (Medio): 0.66 CV, 0.67 test
MAX_EXTREME_ERROR_RATE = 0.03   # Económico <-> Premium; champion 1.0 % CV, 1.2 % test
MAX_GAP = 0.05                  # agreed overfitting limit (train - validation F1)
MIN_MARGIN_OVER_DUMMY = 0.30    # champion is 0.60 above the most-frequent baseline


def quality_problems(
    f1_macro: float,
    class_f1: dict,
    extreme_error_rate: float,
    *,
    gap: float | None = None,
    dummy_f1_macro: float | None = None,
) -> list[str]:
    """Return the minimums that are not met (empty list if the model is acceptable).

    ``gap`` and ``dummy_f1_macro`` are optional because a single test
    evaluation has no train/validation gap or Dummy reference.
    """
    # F1 scores and rates are proportions in [0, 1]; the gap (train - validation
    # F1) is a difference of two of them, so it lies in [-1, 1].
    proportions = [f1_macro, extreme_error_rate, *class_f1.values()]
    if dummy_f1_macro is not None:
        proportions.append(dummy_f1_macro)
    values = proportions + ([gap] if gap is not None else [])
    # bool is a subclass of int in Python: True would be read as a perfect 1.0.
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in values):
        return ["métricas no numéricas o no finitas"]
    if not all(0 <= v <= 1 for v in proportions) or (gap is not None and not -1 <= gap <= 1):
        return ["métricas fuera de rango (F1 y tasas entre 0 y 1, gap entre -1 y 1)"]
    if set(class_f1) != set(TARGET_LABELS):
        return [f"faltan clases en el F1 por clase: {sorted(set(TARGET_LABELS) - set(class_f1))}"]

    problems = []
    if f1_macro < MIN_F1_MACRO:
        problems.append(f"F1 macro {f1_macro:.3f} < {MIN_F1_MACRO}")
    for label in TARGET_LABELS:
        if class_f1[label] < MIN_CLASS_F1:
            problems.append(f"F1 de {label} {class_f1[label]:.3f} < {MIN_CLASS_F1}")
    if extreme_error_rate > MAX_EXTREME_ERROR_RATE:
        problems.append(f"errores extremos {extreme_error_rate:.1%} > {MAX_EXTREME_ERROR_RATE:.0%}")
    if gap is not None and not gap < MAX_GAP:
        problems.append(f"gap {gap:.3f} ≥ {MAX_GAP}")
    if dummy_f1_macro is not None and f1_macro - dummy_f1_macro < MIN_MARGIN_OVER_DUMMY:
        problems.append(
            f"solo {f1_macro - dummy_f1_macro:.3f} por encima de Dummy (< {MIN_MARGIN_OVER_DUMMY})"
        )
    return problems
