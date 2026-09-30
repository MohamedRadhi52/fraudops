import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudops.metrics import auc_pr_of_days, bootstrap_interval, card_precision


def test_card_precision_ranks_cards_and_blocks_the_frauds_found():
    scored = pd.DataFrame(
        {
            "tx_time_days": [0, 0, 0, 0, 1, 1, 1, 1],
            "customer_id": ["A", "A", "B", "C", "A", "B", "D", "C"],
            "score": [0.2, 0.9, 0.8, 0.1, 0.99, 0.5, 0.4, 0.3],
            "tx_fraud": [0, 1, 0, 1, 1, 0, 0, 1],
        }
    )
    # Day 0: A (best score 0.9, fraud) and B are investigated. Day 1: A is blocked.
    assert card_precision(scored, "score", k=2).tolist() == [0.5, 0.0]


def test_auc_pr_of_days_matches_scikit_learn(small_transactions):
    scored = small_transactions.assign(
        score=np.random.default_rng(0).random(len(small_transactions))
    )
    auc_pr = auc_pr_of_days(scored, "score")
    days = np.array([3, 3, 10, 20])
    rows = pd.concat([scored[scored["tx_time_days"] == day] for day in days])
    assert auc_pr(days) == pytest.approx(average_precision_score(rows["tx_fraud"], rows["score"]))


def test_bootstrap_interval_surrounds_the_estimate():
    values = pd.Series(np.random.default_rng(0).normal(10, 1, 50))
    low, high = bootstrap_interval(lambda days: values[days].mean(), values.index.to_numpy())
    assert low < values.mean() < high
    assert high - low < 1
