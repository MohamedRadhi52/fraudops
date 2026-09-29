import pandas as pd
import pytest

from fraudops.simulate import SECONDS_PER_DAY, generate_dataset


@pytest.fixture(scope="module")
def book_dataset():
    return generate_dataset()


def tiny(seed):
    return generate_dataset(n_customers=50, n_terminals=200, nb_days=20, radius=10, seed=seed)


def test_book_parameters_give_about_1_8_million_transactions(book_dataset):
    assert 1_700_000 < len(book_dataset) < 1_850_000


def test_fraud_rate_is_below_one_percent_with_the_three_scenarios(book_dataset):
    assert 0.006 < book_dataset["tx_fraud"].mean() < 0.011
    assert set(book_dataset["tx_fraud_scenario"]) == {0, 1, 2, 3}


def test_same_seed_gives_the_same_dataset():
    pd.testing.assert_frame_equal(tiny(seed=3), tiny(seed=3))
    assert not tiny(seed=3).equals(tiny(seed=4))


def test_transactions_are_consistent(small_transactions):
    tx = small_transactions
    assert (tx["transaction_id"] == range(len(tx))).all()
    assert tx["tx_datetime"].is_monotonic_increasing
    assert (tx["tx_time_days"] == tx["tx_time_seconds"] // SECONDS_PER_DAY).all()
    assert (tx["tx_amount"] >= 0).all()


def test_labels_follow_the_scenarios(small_transactions):
    tx = small_transactions
    assert (tx["tx_fraud"] == (tx["tx_fraud_scenario"] > 0)).all()
    assert (tx.loc[tx["tx_amount"] > 220, "tx_fraud"] == 1).all()
