"""Prequential evaluation of the card models: out-of-sample scores, metrics and figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudops import metrics, models
from fraudops.features import FEATURES_PATH
from fraudops.validate import TEST_WEEKS, VALIDATION_WEEKS, split

SCORES_PATH = Path("data/scores.parquet")
REPORT_DIR = Path("reports")
MODELS = {"rules": models.rules, "logistic": models.logistic_regression}
LABELS = {"rules": "Règles métier", "logistic": "Régression logistique"}
KEPT_COLUMNS = ["transaction_id", "tx_time_days", "customer_id", "tx_amount", "tx_fraud"]


def prequential_scores(
    features: pd.DataFrame, weeks: tuple[int, ...] = VALIDATION_WEEKS + TEST_WEEKS
) -> pd.DataFrame:
    """Scores of every model on the given weeks, each model being retrained every week."""
    scored = []
    for start in weeks:
        train, test = split(features, start)
        scores = {name: fit_predict(train, test) for name, fit_predict in MODELS.items()}
        scored.append(test[KEPT_COLUMNS].assign(**scores))
    return pd.concat(scored, ignore_index=True)


def fmt(value: float, low: float, high: float) -> str:
    """Value and interval in French notation, like 0,52 [0,49 ; 0,55]."""
    return f"{value:.2f} [{low:.2f} ; {high:.2f}]".replace(".", ",")


def plot_card_precision(results: dict, ceiling: float, path: Path) -> None:
    """Fraudulent cards found among the 100 investigated each day, for each model."""
    labels = [LABELS[name] for name in results]
    value, low, high = 100 * np.array([result["card_precision"] for result in results.values()]).T
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.5 * len(labels)))
    ax.barh(labels, value, xerr=[value - low, high - value], color="#4C72B0", capsize=4)
    for y, v in enumerate(value):
        ax.text(0.5, y, f"{v:.0f}", va="center", color="white", weight="bold")
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


def main() -> None:
    scores = prequential_scores(pd.read_parquet(FEATURES_PATH))
    scores["period"] = np.where(scores["tx_time_days"] < TEST_WEEKS[0], "validation", "test")
    scores.to_parquet(SCORES_PATH, index=False)

    test = scores[scores["period"] == "test"]
    results = {name: metrics.summarize(test, name) for name in MODELS}
    # A perfect model scores 1 on frauds: its Card Precision is the best achievable.
    ceiling = metrics.card_precision(test.assign(perfect=test["tx_fraud"]), "perfect").mean()

    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    report = {"models": results, "card_precision_ceiling": ceiling}
    (REPORT_DIR / "models.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_card_precision(results, ceiling, REPORT_DIR / "figures" / "card_precision.png")

    print("| Modèle | Card Precision@100 | AUC-PR |\n|---|---|---|")
    for name, result in results.items():
        cp, auc = fmt(*result["card_precision"]), fmt(*result["auc_pr"])
        print(f"| {LABELS[name]} | {cp} | {auc} |")
    print(f"| Plafond (modèle parfait) | {ceiling:.2f} | 1 |".replace(".", ","))


if __name__ == "__main__":
    main()
