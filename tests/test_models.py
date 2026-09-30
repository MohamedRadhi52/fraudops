import pandas as pd

from fraudops.models import logistic_regression, rules
from fraudops.validate import split


def test_rules_rank_by_number_of_rules_then_amount():
    test = pd.DataFrame(
        {
            "tx_amount": [250.0, 30.0, 40.0, 35.0],
            "terminal_risk_7d": [0.0, 0.5, 0.0, 0.0],
            "customer_avg_amount_30d": [50.0, 30.0, 40.0, 35.0],
        }
    )
    scores = rules(None, test)
    # Two rules (amount above 220 and far above habit), then one rule, then amounts.
    assert list(scores.argsort()[::-1]) == [0, 1, 2, 3]


def test_logistic_regression_gives_probabilities(small_features):
    train, test = split(small_features, test_start=45)
    scores = logistic_regression(train, test)
    assert len(scores) == len(test)
    assert ((scores >= 0) & (scores <= 1)).all()
