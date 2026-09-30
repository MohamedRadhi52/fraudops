"""Prequential validation: train on past weeks, wait for their labels, test on the next week."""

import pandas as pd

from fraudops.features import DELAY

TRAIN_DAYS = 28
TEST_DAYS = 7
# first day of each test week (validation weeks for tuning, test weeks for the results)
VALIDATION_WEEKS = (98, 105, 112, 119)
TEST_WEEKS = (126, 133, 140, 147, 154, 161, 168, 175)


def split(tx: pd.DataFrame, test_start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training window and test week of a model put in production on day `test_start`.

    The training window ends DELAY days before the test week, the time it takes to know its
    labels. In the test week, the transactions of a card are dropped from the day a fraud on
    that card becomes known: the card is blocked.
    """
    day = tx["tx_time_days"]
    train_start = test_start - DELAY - TRAIN_DAYS
    train = tx[(day >= train_start) & (day < test_start - DELAY)]
    test = tx[(day >= test_start) & (day < test_start + TEST_DAYS)]

    frauds = tx[(tx["tx_fraud"] == 1) & (day >= train_start)]
    known_from = frauds.groupby("customer_id")["tx_time_days"].min() + DELAY
    blocked = test["customer_id"].map(known_from) <= test["tx_time_days"]
    return train, test[~blocked]
