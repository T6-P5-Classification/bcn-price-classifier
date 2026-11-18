"""Render the real panel against the shared temporary CSV store."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.cleaning import MODEL_FEATURES
from src.feedback_store import StorageError, save_feedback, save_prediction

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

PANEL = "from feedback_panel import render_feedback_panel\nrender_feedback_panel()"


def prediction():
    return save_prediction({name: 0 for name in MODEL_FEATURES}, "Medio",
                           {"Económico": 0.2, "Medio": 0.6, "Premium": 0.2},
                           model_name="synthetic", model_version="v1",
                           thresholds={"lower": 122., "upper": 250.})


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path))
    return AppTest.from_string(PANEL, default_timeout=20)


def test_empty_panel_does_not_show_fabricated_scores(panel):
    app = panel.run()
    assert not app.exception
    assert not app.metric
    assert any("No hay predicciones" in item.value for item in app.info)


def test_manual_refresh_reads_new_saved_feedback(panel):
    identifier = prediction()
    app = panel.run()
    assert not app.exception
    assert any("sin respuestas" in item.value for item in app.info)
    assert "F1 macro" not in [item.label for item in app.metric]
    save_feedback(identifier, 200.)
    app.button(key="feedback_refresh").click().run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Respuestas"] == "1"
    assert metrics["Accuracy"] == "100,0 %"
    assert metrics["F1 macro"] == "0,333"
    assert len(app.dataframe) == 3
    assert any("voluntarias" in item.value for item in app.caption)


def test_storage_failure_does_not_expose_details(panel, monkeypatch):
    import feedback_panel as module
    def fail():
        raise StorageError("secret-db-path")
    monkeypatch.setattr(module, "load_labelled", fail)
    app = panel.run()
    assert not app.exception
    assert app.warning
    assert "secret-db-path" not in repr(app)
