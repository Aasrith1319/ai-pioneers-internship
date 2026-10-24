"""Automated tests for the deployed API. Run with:  python -m pytest -q"""
import pytest
from app import app

PASSENGER = {"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0,
             "fare": 211.3, "embarked": "S", "has_deck": 1}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_single_prediction(client):
    r = client.post("/predict", json=PASSENGER)
    assert r.status_code == 200
    assert r.json["survived"] in (0, 1)
    assert 0 <= r.json["survival_probability"] <= 1


def test_batch_prediction_orders_risk_sensibly(client):
    poor_male = {**PASSENGER, "pclass": 3, "sex": "male", "fare": 7.25, "has_deck": 0}
    r = client.post("/predict", json=[PASSENGER, poor_male])
    assert r.status_code == 200 and len(r.json) == 2
    assert r.json[0]["survival_probability"] > r.json[1]["survival_probability"]


def test_validation_errors(client):
    r = client.post("/predict", json={**PASSENGER, "pclass": 5, "sex": "robot"})
    assert r.status_code == 422
    assert len(r.json["details"]["0"]) == 2


def test_missing_field(client):
    bad = dict(PASSENGER); bad.pop("age")
    assert client.post("/predict", json=bad).status_code == 422


def test_non_json(client):
    assert client.post("/predict", data="hello").status_code == 400


def test_html_form(client):
    r = client.post("/", data={**PASSENGER, "has_deck": "1"})
    assert r.status_code == 200 and b"survival probability" in r.data
