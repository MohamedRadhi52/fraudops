"""Metrics on scored test transactions, with bootstrap confidence intervals over days."""

from collections.abc import Callable

import numpy as np
import pandas as pd

CAPACITY = 100  # cards the team can investigate each day
TARGET_FPR = 0.05  # share of legitimate BAF applications that may be flagged
N_RESAMPLES = 1000


def investigations(
    scored: pd.DataFrame, score: str, k: int = CAPACITY, threshold: float = -np.inf
) -> pd.DataFrame:
    """Cards investigated each day: the k most suspicious ones above `threshold`, not blocked yet.

    A card's score is its highest transaction score of the day. The fraudulent cards found by
    the investigation are blocked and leave the following days.
    """
    cards = scored.groupby(["tx_time_days", "customer_id"])[[score, "tx_fraud"]].max()
    blocked = set()
    investigated = []
    for _, day_cards in cards.groupby(level="tx_time_days"):
        customers = day_cards.index.get_level_values("customer_id")
        candidates = day_cards[~customers.isin(blocked) & (day_cards[score] >= threshold)]
        top = candidates.nlargest(k, score)
        investigated.append(top)
        blocked.update(top.index.get_level_values("customer_id")[top["tx_fraud"] == 1])
    return pd.concat(investigated)


def stopped(frauds: pd.DataFrame, investigated: pd.DataFrame) -> pd.Series:
    """Whether each fraud is stopped: its card is investigated that day, or blocked before."""
    found = investigated[investigated["tx_fraud"] == 1].reset_index()
    blocked_from = found.groupby("customer_id")["tx_time_days"].min()
    that_day = pd.MultiIndex.from_frame(frauds[["tx_time_days", "customer_id"]]).isin(
        investigated.index
    )
    return that_day | (frauds["customer_id"].map(blocked_from) < frauds["tx_time_days"])


def card_precision(scored: pd.DataFrame, score: str, k: int = CAPACITY) -> pd.Series:
    """Card Precision@k of each day: share of frauds among the k cards investigated."""
    return investigations(scored, score, k).groupby(level="tx_time_days")["tx_fraud"].sum() / k


def auc_pr_of_days(scored: pd.DataFrame, score: str) -> Callable[[np.ndarray], float]:
    """Function giving the AUC-PR of any sample of days, a day drawn twice counting twice."""
    scored = scored.sort_values(score, ascending=False)
    labels = scored["tx_fraud"].to_numpy()
    day_codes, days = pd.factorize(scored["tx_time_days"])
    # last row of each run of equal scores, where the precision-recall curve moves
    steps = np.flatnonzero(np.append(np.diff(scored[score].to_numpy()) != 0, True))

    def auc_pr(sample: np.ndarray) -> float:
        weights = np.bincount(days.get_indexer(sample), minlength=len(days))[day_codes]
        found = np.cumsum(weights * labels)[steps]
        precision = found / np.maximum(np.cumsum(weights)[steps], 1)
        return float(np.sum(np.diff(found, prepend=0) * precision) / found[-1])

    return auc_pr


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float], units: np.ndarray, seed: int = 0
) -> list[float]:
    """95% interval of the statistic over resamplings of the units (days, applications)."""
    rng = np.random.default_rng(seed)
    samples = [statistic(rng.choice(units, len(units))) for _ in range(N_RESAMPLES)]
    return np.percentile(samples, [2.5, 97.5]).tolist()


def format_interval(value: float, low: float, high: float, percent: bool = False) -> str:
    """Value and 95% interval in French notation: 0,66 [0,65 ; 0,68] or 19,5 % [18,5 ; 20,4]."""
    if percent:
        return f"{100 * value:.1f} % [{100 * low:.1f} ; {100 * high:.1f}]".replace(".", ",")
    return f"{value:.2f} [{low:.2f} ; {high:.2f}]".replace(".", ",")


def summarize(scored: pd.DataFrame, score: str) -> dict[str, list[float]]:
    """Card Precision@100 and AUC-PR, each as [value, low, high]."""
    days = np.sort(scored["tx_time_days"].unique())
    daily = card_precision(scored, score)
    auc_pr = auc_pr_of_days(scored, score)
    return {
        "card_precision": [daily.mean(), *bootstrap_interval(lambda d: daily[d].mean(), days)],
        "auc_pr": [auc_pr(days), *bootstrap_interval(auc_pr, days)],
    }


def paired_difference(scored: pd.DataFrame, score: str, baseline: str) -> list[float]:
    """Card Precision@100 of `score` minus `baseline`, as [value, low, high].

    Both models are compared on the same resampled days (paired bootstrap).
    """
    difference = card_precision(scored, score) - card_precision(scored, baseline)
    days = difference.index.to_numpy()
    return [difference.mean(), *bootstrap_interval(lambda d: difference[d].mean(), days)]


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = TARGET_FPR) -> float:
    """Score above which a share `fpr` of the legitimate cases is flagged."""
    return float(np.quantile(scores[labels == 0], 1 - fpr))


def rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Recall and false positive rate of the cases scoring above `threshold`."""
    flagged = scores > threshold
    return {"recall": flagged[labels == 1].mean(), "fpr": flagged[labels == 0].mean()}


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population stability index of `current` against `reference`.

    Categories and variables with few values are compared value by value, the others on the
    deciles of the reference.
    """
    if isinstance(reference.dtype, pd.CategoricalDtype) or reference.nunique() <= bins:
        expected = reference.value_counts(normalize=True)
        actual = current.value_counts(normalize=True).reindex(expected.index, fill_value=0)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1))[1:-1])
        size = len(edges) + 1
        expected = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=size)
        actual = np.bincount(np.searchsorted(edges, current, side="right"), minlength=size)
        expected, actual = expected / len(reference), actual / len(current)
    # floor to avoid log(0) when a bin is empty
    expected, actual = np.maximum(expected, 1e-4), np.maximum(actual, 1e-4)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
