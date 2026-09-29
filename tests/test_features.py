import pandas as pd
import pytest

from fraudops.features import DELAY, WINDOWS, build_features


@pytest.fixture(scope="module")
def features(small_transactions):
    return build_features(small_transactions)


def naive_features(tx, row):
    """Features of one transaction, recomputed from their definition."""
    t = row.tx_datetime
    customer = tx[(tx["customer_id"] == row.customer_id) & (tx["tx_datetime"] <= t)]
    terminal = tx[tx["terminal_id"] == row.terminal_id]
    known = terminal["tx_datetime"] <= t - pd.Timedelta(days=DELAY)
    expected = {}
    for days in WINDOWS:
        recent = customer[customer["tx_datetime"] > t - pd.Timedelta(days=days)]
        window = terminal[known & (terminal["tx_datetime"] > t - pd.Timedelta(days=DELAY + days))]
        expected[f"customer_nb_tx_{days}d"] = len(recent)
        expected[f"customer_avg_amount_{days}d"] = recent["tx_amount"].mean()
        expected[f"terminal_nb_tx_{days}d"] = len(window)
        expected[f"terminal_risk_{days}d"] = window["tx_fraud"].mean() if len(window) else 0.0
    return expected


def test_features_match_their_definition(small_transactions, features):
    for row in features.sample(200, random_state=0).itertuples():
        for name, value in naive_features(small_transactions, row).items():
            assert getattr(row, name) == pytest.approx(value), name


def test_features_ignore_future_transactions(small_transactions, features):
    cutoff = small_transactions["tx_datetime"].quantile(0.5)
    past = small_transactions[small_transactions["tx_datetime"] <= cutoff]
    pd.testing.assert_frame_equal(build_features(past), features.loc[past.index])


def test_features_ignore_labels_not_known_yet(small_transactions, features):
    tx = small_transactions.copy()
    cutoff = tx["tx_datetime"].quantile(0.5)
    unknown = tx["tx_datetime"] > cutoff - pd.Timedelta(days=DELAY)
    tx.loc[unknown, "tx_fraud"] = 1 - tx.loc[unknown, "tx_fraud"]
    flipped = build_features(tx)

    columns = features.columns.difference(tx.columns)
    scored = tx["tx_datetime"] <= cutoff
    pd.testing.assert_frame_equal(flipped.loc[scored, columns], features.loc[scored, columns])
    # The flipped labels do change the features of later transactions: the test can fail.
    assert not flipped.loc[~scored, columns].equals(features.loc[~scored, columns])


def test_transactions_of_the_same_second_see_each_other():
    tx = pd.DataFrame(
        {
            "tx_datetime": pd.to_datetime(
                ["2018-04-01 10:00", "2018-04-01 12:00", "2018-04-01 12:00"]
            ),
            "customer_id": [0, 0, 0],
            "terminal_id": [0, 1, 2],
            "tx_amount": [10.0, 20.0, 30.0],
            "tx_fraud": [0, 0, 0],
        }
    )
    features = build_features(tx)
    assert features["customer_nb_tx_1d"].tolist() == [1, 3, 3]
    assert features["customer_avg_amount_1d"].tolist() == [10.0, 20.0, 20.0]
