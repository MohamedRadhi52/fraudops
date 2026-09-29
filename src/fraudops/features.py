"""Rolling-window features on customers and terminals, computed with pandas."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.typing import RollingGroupby

from fraudops.simulate import TRANSACTIONS_PATH

FEATURES_PATH = Path("data/features.parquet")
WINDOWS = (1, 7, 30)  # days
DELAY = 7  # days before a fraud label is known


def rolling(tx: pd.DataFrame, key: str, column: str, days: int) -> RollingGroupby:
    """Window over the last `days` days of each key. `tx` must be sorted by key and time."""
    return tx.groupby(key)[["tx_datetime", column]].rolling(f"{days}D", on="tx_datetime")


def customer_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Number and mean amount of the customer's transactions over the last 1, 7 and 30 days.

    Each window ends with the scored transaction: its amount is known when it is scored.
    """
    tx = tx.sort_values(["customer_id", "tx_datetime"])
    columns = {}
    for days in WINDOWS:
        window = rolling(tx, "customer_id", "tx_amount", days)
        columns[f"customer_nb_tx_{days}d"] = window.count()["tx_amount"].astype(np.int64)
        columns[f"customer_avg_amount_{days}d"] = window.mean()["tx_amount"]
    features = pd.DataFrame(columns).droplevel(0)
    # pandas ends a window at the current row: extend it to the customer's other
    # transactions of the same second, like any time-based window.
    return features.groupby([tx["customer_id"], tx["tx_datetime"]]).transform("last")


def terminal_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Number of transactions and fraud rate of the terminal over 1, 7 and 30 days.

    A label is only known DELAY days after the transaction, so each window ends DELAY days
    before the scored transaction.
    """
    tx = tx.sort_values(["terminal_id", "tx_datetime"])

    def count_and_frauds(days: int) -> tuple[pd.Series, pd.Series]:
        window = rolling(tx, "terminal_id", "tx_fraud", days)
        return window.count()["tx_fraud"], window.sum()["tx_fraud"]

    unknown_count, unknown_frauds = count_and_frauds(DELAY)
    columns = {}
    for days in WINDOWS:
        count, frauds = count_and_frauds(DELAY + days)
        count, frauds = count - unknown_count, frauds - unknown_frauds
        columns[f"terminal_nb_tx_{days}d"] = count.astype(np.int64)
        columns[f"terminal_risk_{days}d"] = (frauds / count).fillna(0.0)
    return pd.DataFrame(columns).droplevel(0)


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Transactions with their customer and terminal features."""
    return tx.join(customer_features(tx)).join(terminal_features(tx))


def main() -> None:
    tx = pd.read_parquet(TRANSACTIONS_PATH)
    features = build_features(tx)
    features.to_parquet(FEATURES_PATH, index=False)
    n_features = features.shape[1] - tx.shape[1]
    print(f"{n_features} features for {len(tx):,} transactions, saved to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
