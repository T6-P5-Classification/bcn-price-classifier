"""Static checks of docker-compose.yml (#26); starting the stack is checked by hand."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_app_and_database_services():
    assert set(COMPOSE["services"]) == {"app", "mongo"}
    app = COMPOSE["services"]["app"]
    assert app["build"] == "." and "8501:8501" in app["ports"]
    assert app["depends_on"]["mongo"]["condition"] == "service_healthy"


def test_feedback_survives_restarts_and_stays_out_of_the_image():
    app = COMPOSE["services"]["app"]
    assert app["environment"]["FEEDBACK_DIR"] == "/app/data/feedback"
    assert "feedback-data:/app/data/feedback" in app["volumes"]
    assert {"feedback-data", "mongo-data"} <= set(COMPOSE["volumes"])


def test_database_is_pinned_private_and_checked():
    mongo = COMPOSE["services"]["mongo"]
    tag = mongo["image"].split(":")[1]
    assert tag.count(".") == 2  # exact version, not "latest" or "7"
    assert "ports" not in mongo  # only reachable from the app, inside Compose
    assert "mongo-data:/data/db" in mongo["volumes"]
    assert "healthcheck" in mongo
    host = COMPOSE["services"]["app"]["environment"]["MONGODB_URI"]
    assert host.startswith("mongodb://mongo:27017/")
