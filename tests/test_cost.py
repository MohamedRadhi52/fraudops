import numpy as np
import pandas as pd
import pytest

from fraudops.cost import CONTROL_COST, daily_cost


@pytest.fixture
def scored():
    return pd.DataFrame(
        {
            "tx_time_days": [0, 0, 0, 1, 1, 1],
            "customer_id": ["A", "B", "C", "A", "C", "D"],
            "score": [0.9, 0.2, 0.6, 0.95, 0.1, 0.3],
            "tx_fraud": [1, 0, 1, 1, 1, 1],
            "tx_amount": [100.0, 20.0, 50.0, 80.0, 30.0, 40.0],
        }
    )


def test_daily_cost_adds_controls_and_missed_frauds(scored):
    daily = daily_cost(scored, "score", threshold=0.5)
    # Day 0: A and C are investigated and blocked. Day 1: no control, their frauds are stopped
    # anyway, but the fraud of D, below the threshold, is missed.
    assert daily["controls"].tolist() == [2, 0]
    assert daily["cost"].tolist() == [2 * CONTROL_COST, 40.0]


def test_without_control_every_fraud_is_lost(scored):
    assert daily_cost(scored, "score", threshold=np.inf)["cost"].sum() == 300.0
