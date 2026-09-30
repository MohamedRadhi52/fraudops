import pandas as pd

from fraudops.features import DELAY
from fraudops.validate import TEST_DAYS, TRAIN_DAYS, split


def test_training_window_ends_delay_days_before_the_test_week(small_features):
    train, test = split(small_features, test_start=45)
    assert train["tx_time_days"].min() == 45 - DELAY - TRAIN_DAYS
    assert train["tx_time_days"].max() == 45 - DELAY - 1
    assert test["tx_time_days"].between(45, 45 + TEST_DAYS - 1).all()


def test_a_card_is_blocked_once_its_fraud_is_known():
    tx = pd.DataFrame(
        {
            "tx_time_days": [10, 15, 16, 17, 18, 17],
            "customer_id": [1, 1, 1, 1, 1, 2],
            "tx_fraud": [1, 0, 0, 1, 0, 0],
        }
    )
    _, test = split(tx, test_start=15)
    # The fraud of day 10 is known on day 17: card 1 is blocked from that day.
    assert test[["tx_time_days", "customer_id"]].values.tolist() == [[15, 1], [16, 1], [17, 2]]
