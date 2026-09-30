"""Metrics on scored test transactions, with bootstrap confidence intervals over days."""

from collections.abc import Callable

import numpy as np
import pandas as pd

CAPACITY = 100  # cards the team can investigate each day
N_RESAMPLES = 1000


def card_precision(scored: pd.DataFrame, score: str, k: int = CAPACITY) -> pd.Series:
    """Card Precision@k of each day: share of frauds among the k most suspicious cards.

    A card's score is its highest transaction score of the day. The fraudulent cards found by
    the investigation are blocked and leave the following days.
    """
    cards = scored.groupby(["tx_time_days", "customer_id"])[[score, "tx_fraud"]].max()
    blocked = set()
    precision = {}
    for day, day_cards in cards.groupby(level="tx_time_days"):
        day_cards = day_cards.droplevel("tx_time_days")
        top = day_cards[~day_cards.index.isin(blocked)].nlargest(k, score)
        found = top.index[top["tx_fraud"] == 1]
        precision[day] = len(found) / k
        blocked.update(found)
    return pd.Series(precision)


def auc_pr_of_days(scored: pd.DataFrame, score: str) -> Callable[[np.ndarray], float]:
    """Function giving the AUC-PR of any sample of days, a day drawn twice counting twice."""
    scored = scored.sort_values(score, ascending=False)
    labels = scored["tx_fraud"].to_numpy()
    day_codes, days = pd.factorize(scored["tx_time_days"])
    # Last row of each run of equal scores: the precision-recall curve moves there.
    steps = np.flatnonzero(np.append(np.diff(scored[score].to_numpy()) != 0, True))

    def auc_pr(sample: np.ndarray) -> float:
        weights = np.bincount(days.get_indexer(sample), minlength=len(days))[day_codes]
        found = np.cumsum(weights * labels)[steps]
        precision = found / np.maximum(np.cumsum(weights)[steps], 1)
        return float(np.sum(np.diff(found, prepend=0) * precision) / found[-1])

    return auc_pr


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float], days: np.ndarray, seed: int = 0
) -> list[float]:
    """95% interval of the statistic over resamplings of the days, drawn with replacement."""
    rng = np.random.default_rng(seed)
    samples = [statistic(rng.choice(days, len(days))) for _ in range(N_RESAMPLES)]
    return np.percentile(samples, [2.5, 97.5]).tolist()


def summarize(scored: pd.DataFrame, score: str) -> dict[str, list[float]]:
    """Card Precision@100 and AUC-PR, each as [value, low, high]."""
    days = np.sort(scored["tx_time_days"].unique())
    daily = card_precision(scored, score)
    auc_pr = auc_pr_of_days(scored, score)
    return {
        "card_precision": [daily.mean(), *bootstrap_interval(lambda d: daily[d].mean(), days)],
        "auc_pr": [auc_pr(days), *bootstrap_interval(auc_pr, days)],
    }
