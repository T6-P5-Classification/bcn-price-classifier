"""Regression check: the model notebook must not inspect held-out data."""
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.cleaning import MODEL_FEATURES
from src.preprocessing import TARGET_LABELS


class ForbiddenHoldout:
    def __getattr__(self, name):
        raise AssertionError("Held-out data must not be inspected in the model notebook")

    def __contains__(self, value):
        raise AssertionError("Held-out data must not be inspected in the model notebook")

    def __len__(self):
        raise AssertionError("Held-out data must not be inspected in the model notebook")

    def __getitem__(self, key):
        raise AssertionError("Held-out data must not be inspected in the model notebook")


def test_logreg_diagnostics_use_train_oof_without_accessing_holdout(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MPLCONFIGDIR", str(root / ".jupyter" / "matplotlib"))
    import src.preprocessing as shared
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook = json.loads((root / "notebooks/04_model_logreg.ipynb").read_text())
    group = np.repeat(np.arange(3), 40)
    rows = pd.DataFrame({c: group.astype(float) + 1 for c in MODEL_FEATURES})
    for column in ("neighbourhood_cleansed", "neighbourhood_group_cleansed", "property_type", "room_type"):
        rows[column] = [f"synthetic_{g}" for g in group]
    target = pd.Series(np.repeat(TARGET_LABELS, 40), index=rows.index)
    holdout = ForbiddenHoldout()
    monkeypatch.setattr(shared, "load_train_test", lambda **kwargs: (
        rows, holdout, target, holdout, {"lower": 10.0, "upper": 20.0}
    ))
    monkeypatch.chdir(root)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    namespace = {}
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            # Export/provenance is checked in the real end-to-end run. This
            # test exercises data loading, real CV/Optuna and diagnostics.
            source = "".join(cell["source"])
            if "OUT = ROOT" in source:
                break
            exec(compile(source, "04_model_logreg.ipynb", "exec"), namespace)
            # Keep regression checks quick; the delivered run uses 20 trials.
            namespace["N_TRIALS"] = 2
            namespace["display"] = lambda *args, **kwargs: None
        # cross_val_predict fits clones, leaving the selected Pipeline unfitted.
        # A regression to fit-all-train diagnostics fails this check.
        with pytest.raises(NotFittedError):
            check_is_fitted(namespace["selected_model"])
        comparison = namespace["comparison"]
        assert set(comparison["evaluation_set"]) == {"train_oof"}
        assert not any(c.startswith("test_") for c in comparison.columns)
        assert set(comparison["model"]) == {"DummyClassifier", "LogisticRegression"}
        for model, matrix in namespace["matrices"].items():
            assert np.asarray(matrix).sum() == len(target)
            assert namespace["reports"][model]["macro avg"]["support"] == len(target)
    finally:
        plt.close("all")
