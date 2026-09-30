import json
from pathlib import Path

from fastapi.testclient import TestClient

from fraudops import cost
from fraudops.api import CONTROL_COST, app

client = TestClient(app)
# A fraud of the last week: 913.60 euros for a customer spending 216 euros on average.
FRAUD = json.loads(Path("tests/transaction.json").read_text())


def test_a_fraud_is_investigated_with_three_reasons():
    body = client.post("/score", json=FRAUD).json()
    assert body["probability"] > 0.9
    assert body["decision"] == "contrôler"
    assert [reason["feature"] for reason in body["reasons"]][0] == "tx_amount"
    assert len(body["reasons"]) == 3


def test_an_ordinary_payment_goes_through():
    ordinary = FRAUD | {
        "tx_amount": 25.0,
        "customer_avg_amount_1d": 25.0,
        "customer_avg_amount_7d": 27.0,
        "customer_avg_amount_30d": 26.0,
    }
    assert client.post("/score", json=ordinary).json()["decision"] == "laisser passer"


def test_invalid_transactions_are_rejected():
    assert client.post("/score", json=FRAUD | {"tx_amount": -5}).status_code == 422
    assert client.post("/score", json=FRAUD | {"terminal_risk_7d": 2}).status_code == 422


def test_the_api_uses_the_control_cost_of_the_study():
    assert CONTROL_COST == cost.CONTROL_COST
