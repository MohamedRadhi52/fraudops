"""Models of the card dataset. Each one trains on a window and scores the next week."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudops.features import WINDOWS

FEATURES = [
    "tx_amount",
    *[f"customer_{stat}_{days}d" for days in WINDOWS for stat in ("nb_tx", "avg_amount")],
    *[f"terminal_{stat}_{days}d" for days in WINDOWS for stat in ("nb_tx", "risk")],
]
LIGHTGBM_GRID = [
    {"num_leaves": leaves, "learning_rate": rate} for leaves in (15, 31, 63) for rate in (0.05, 0.1)
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
    # amount / 10 000 stays below 1, so it only breaks ties between rule counts
    return (triggered + test["tx_amount"] / 10_000).to_numpy()


def logistic_regression(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fraud probability from a logistic regression on standardized features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train["tx_fraud"])
    return model.predict_proba(test[FEATURES])[:, 1]


def fit_lightgbm(train: pd.DataFrame, params: dict) -> LGBMClassifier:
    """LightGBM stopped early on the last week of the training window."""
    last_week = train["tx_time_days"] > train["tx_time_days"].max() - 7
    fit, stop = train[~last_week], train[last_week]
    # deterministic so the results don't depend on the number of threads
    model = LGBMClassifier(
        n_estimators=1000, deterministic=True, force_row_wise=True, verbose=-1, **params
    )
    return model.fit(
        fit[FEATURES],
        fit["tx_fraud"],
        eval_X=stop[FEATURES],
        eval_y=stop["tx_fraud"],
        eval_metric="average_precision",
        callbacks=[early_stopping(50, verbose=False)],
    )


def lightgbm(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> np.ndarray:
    """Fraud probability from LightGBM trained on the window."""
    return fit_lightgbm(train, params).predict_proba(test[FEATURES])[:, 1]
