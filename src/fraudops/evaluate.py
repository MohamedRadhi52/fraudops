"""Prequential evaluation of the card models: out-of-sample scores, metrics and figure."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from fraudops import metrics, models
from fraudops.features import FEATURES_PATH
from fraudops.validate import TEST_WEEKS, VALIDATION_WEEKS, split

SCORES_PATH = Path("data/scores.parquet")
REPORT_DIR = Path("reports")
BASELINES = {"rules": models.rules, "logistic": models.logistic_regression}
LABELS = {"rules": "Règles métier", "logistic": "Régression logistique", "lightgbm": "LightGBM"}
KEPT_COLUMNS = [
    "transaction_id",
    "tx_time_days",
    "customer_id",
    "tx_amount",
    "tx_fraud",
    "tx_fraud_scenario",
    "terminal_risk_30d",
]


def prequential_scores(
    features: pd.DataFrame,
    fit_predicts: dict[str, Callable],
    weeks: tuple[int, ...] = VALIDATION_WEEKS + TEST_WEEKS,
) -> pd.DataFrame:
    """Scores of each model on the given weeks, every model being retrained each week."""
    scored = []
    for start in weeks:
        train, test = split(features, start)
        scores = {name: fit_predict(train, test) for name, fit_predict in fit_predicts.items()}
        scored.append(test[KEPT_COLUMNS].assign(**scores))
    return pd.concat(scored, ignore_index=True)


def tune_lightgbm(features: pd.DataFrame) -> dict:
    """Hyperparameters with the best AUC-PR on the validation weeks, each run logged in MLflow."""
    results = []
    for params in models.LIGHTGBM_GRID:
        fit_predict = partial(models.lightgbm, params=params)
        scored = prequential_scores(features, {"lightgbm": fit_predict}, VALIDATION_WEEKS)
        auc_pr = metrics.auc_pr_of_days(scored, "lightgbm")(scored["tx_time_days"].unique())
        card_precision = metrics.card_precision(scored, "lightgbm").mean()
        with mlflow.start_run(run_name="lightgbm_tuning"):
            mlflow.log_params(params)
            mlflow.log_metrics(
                {"validation_auc_pr": auc_pr, "validation_card_precision": card_precision}
            )
        results.append((auc_pr, params))
    return max(results, key=lambda result: result[0])[1]


def share_stopped(test: pd.DataFrame, score: str) -> dict[str, float]:
    """Share of frauds stopped by the investigations, by scenario.

    Scenario 2 is split: frauds on a terminal with a fraud already known, and the others.
    """
    frauds = test[test["tx_fraud"] == 1]
    stopped = metrics.stopped(frauds, metrics.investigations(test, score))
    group = "scenario " + frauds["tx_fraud_scenario"].astype(str)
    unknown_terminal = (frauds["tx_fraud_scenario"] == 2) & (frauds["terminal_risk_30d"] == 0)
    group = group.where(~unknown_terminal, "scenario 2, terminal without known fraud")
    return stopped.groupby(group).mean().to_dict()


def format_interval(value: float, low: float, high: float, percent: bool = False) -> str:
    """Value and 95% interval in French notation: 0,66 [0,65 ; 0,68] or 19,5 % [18,5 ; 20,4]."""
    if percent:
        return f"{100 * value:.1f} % [{100 * low:.1f} ; {100 * high:.1f}]".replace(".", ",")
    return f"{value:.2f} [{low:.2f} ; {high:.2f}]".replace(".", ",")


def plot_card_precision(results: dict, ceiling: float, path: Path) -> None:
    """Fraudulent cards found among the 100 investigated each day, for each model."""
    labels = [LABELS[name] for name in results]
    value, low, high = 100 * np.array([result["card_precision"] for result in results.values()]).T
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.5 * len(labels)))
    ax.barh(labels, value, xerr=[value - low, high - value], color="#4C72B0", capsize=4)
    for y, v in enumerate(value):
        ax.text(0.5, y, f"{v:.1f}".replace(".", ","), va="center", color="white", weight="bold")
    ax.axvline(100 * ceiling, color="#C44E52", linestyle="--")
    ax.text(
        100 * ceiling,
        1.02,
        f"modèle parfait : {100 * ceiling:.0f}",
        color="#C44E52",
        ha="center",
        transform=ax.get_xaxis_transform(),
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 110 * ceiling)
    ax.set_xlabel(
        "Cartes frauduleuses parmi les 100 contrôlées chaque jour (56 jours de test, IC 95 %)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(report: dict) -> None:
    print("| Modèle | Card Precision@100 | AUC-PR |\n|---|---|---|")
    for name, result in report["models"].items():
        card_precision = format_interval(*result["card_precision"], percent=True)
        print(f"| {LABELS[name]} | {card_precision} | {format_interval(*result['auc_pr'])} |")
    ceiling = 100 * report["card_precision_ceiling"]
    print(f"| Plafond (modèle parfait) | {ceiling:.1f} % | 1 |".replace(".", ","))
    for name, gain in report["lightgbm_card_precision_gain"].items():
        print(f"LightGBM moins {LABELS[name]} : {format_interval(*gain, percent=True)}")
    for group, share in report["lightgbm_share_of_frauds_stopped"].items():
        print(f"Fraudes stoppées avec LightGBM, {group} : {share:.0%}")


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    mlflow.set_experiment("fraudops")
    params = tune_lightgbm(features)
    fit_predicts = BASELINES | {"lightgbm": partial(models.lightgbm, params=params)}
    scores = prequential_scores(features, fit_predicts)
    scores["period"] = np.where(scores["tx_time_days"] < TEST_WEEKS[0], "validation", "test")
    scores.to_parquet(SCORES_PATH, index=False)

    test = scores[scores["period"] == "test"]
    results = {name: metrics.summarize(test, name) for name in fit_predicts}
    # A perfect model scores 1 on frauds: its Card Precision is the best achievable.
    ceiling = metrics.card_precision(test.assign(perfect=test["tx_fraud"]), "perfect").mean()
    gains = {name: metrics.paired_difference(test, "lightgbm", name) for name in BASELINES}
    for name, result in results.items():
        with mlflow.start_run(run_name=name):
            if name == "lightgbm":
                mlflow.log_params(params)
            mlflow.log_metrics({f"test_{metric}": value[0] for metric, value in result.items()})

    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    report = {
        "models": results,
        "card_precision_ceiling": ceiling,
        "lightgbm_params": params,
        "lightgbm_card_precision_gain": gains,
        "lightgbm_share_of_frauds_stopped": share_stopped(test, "lightgbm"),
    }
    (REPORT_DIR / "models.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_card_precision(results, ceiling, REPORT_DIR / "figures" / "card_precision.png")

    print_summary(report)


if __name__ == "__main__":
    main()
