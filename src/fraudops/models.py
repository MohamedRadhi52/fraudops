"""Models of the card dataset. Each one trains on a window and scores the next week."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudops.features import WINDOWS

FEATURES = [
    "tx_amount",
    *[f"customer_{stat}_{days}d" for days in WINDOWS for stat in ("nb_tx", "avg_amount")],
    *[f"terminal_{stat}_{days}d" for days in WINDOWS for stat in ("nb_tx", "risk")],
]


def rules(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Alert rules from the exploration findings, chosen on the validation weeks.

    Nothing is learned. Cards are ranked by number of rules triggered, then by amount.
    """
    triggered = (
        (test["tx_amount"] > 220).astype(int)
        + (test["terminal_risk_7d"] > 0)
        + (test["tx_amount"] > 3 * test["customer_avg_amount_30d"])
    )
    # Amounts stay far below 10,000 euros: they only break ties between rule counts.
    return (triggered + test["tx_amount"] / 10_000).to_numpy()


def logistic_regression(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fraud probability from a logistic regression on standardized features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train["tx_fraud"])
    return model.predict_proba(test[FEATURES])[:, 1]
