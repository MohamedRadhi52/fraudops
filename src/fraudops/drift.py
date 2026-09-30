"""Drift monitoring with the population stability index (PSI)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from fraudops.evaluate import REPORT_DIR, SCORES_PATH
from fraudops.features import FEATURES_PATH
from fraudops.metrics import psi
from fraudops.models import FEATURES, fit_lightgbm
from fraudops.validate import VALIDATION_WEEKS, split

WATCH = 0.1  # usual alert levels of the PSI
RETRAIN = 0.25


def psi_table(
    reference: pd.DataFrame, periods: dict[str, pd.DataFrame], columns: list[str]
) -> pd.DataFrame:
    """PSI of each column (rows) in each period (columns), against the reference."""
    return pd.DataFrame(
        {
            label: {column: psi(reference[column], period[column]) for column in columns}
            for label, period in periods.items()
        }
    )


def status(value: float) -> str:
    """What to do for a PSI value, with the usual alert levels."""
    if value >= RETRAIN:
        return "ré-entraîner"
    if value >= WATCH:
        return "surveiller"
    return "stable"


def summary_table(table: pd.DataFrame, score: str) -> str:
    """Markdown table: PSI of the score and number of variables past each alert level."""
    lines = [
        "| Période | PSI du score | Variables à surveiller | Variables à ré-entraîner |",
        "|---|---|---|---|",
    ]
    variables = table.drop(index=score)
    for period in table.columns:
        watch = int(variables[period].between(WATCH, RETRAIN, inclusive="left").sum())
        retrain = int((variables[period] >= RETRAIN).sum())
        value = f"{table.loc[score, period]:.3f}".replace(".", ",")
        lines.append(f"| {period} | {value} | {watch} | {retrain} |")
    return "\n".join(lines) + "\n"


def plot_psi(table: pd.DataFrame, path, top: int = 10) -> None:
    """Heatmap of the most drifting rows, colored by alert level."""
    table = table.loc[table.max(axis=1).nlargest(top).index]
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * table.shape[1], 1 + 0.4 * len(table)))
    levels = np.digitize(table.to_numpy(), [WATCH, RETRAIN])
    colors = ListedColormap(["#E8EEF7", "#F6C85F", "#D1495B"])
    ax.imshow(levels, cmap=colors, vmin=0, vmax=2, aspect="auto")
    for (i, j), value in np.ndenumerate(table.to_numpy()):
        ax.text(j, i, f"{value:.2f}".replace(".", ","), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    ax.tick_params(length=0)
    ax.set_title(
        f"PSI : jaune au-delà de {WATCH}, rouge au-delà de {RETRAIN}".replace(".", ","),
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Weekly PSI of the card features and score, against the validation weeks."""
    features = pd.read_parquet(FEATURES_PATH)
    scores = pd.read_parquet(SCORES_PATH, columns=["transaction_id", "period"])
    scores = scores.merge(features, on="transaction_id")
    # frozen model, weekly retraining alone would shift the scale of the scores
    params = json.loads((REPORT_DIR / "models.json").read_text())["lightgbm_params"]
    train, _ = split(features, VALIDATION_WEEKS[0])
    scores["score LightGBM"] = fit_lightgbm(train, params).predict_proba(scores[FEATURES])[:, 1]
    test = scores[scores["period"] == "test"]
    weeks = {
        f"semaine {number}": week
        for number, (_, week) in enumerate(test.groupby(test["tx_time_days"] // 7), start=1)
    }
    table = psi_table(
        scores[scores["period"] == "validation"], weeks, [*FEATURES, "score LightGBM"]
    )

    report = {"levels": {"watch": WATCH, "retrain": RETRAIN}, "weekly_psi": table.to_dict()}
    (REPORT_DIR / "drift.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PSI maximal : {table.max().max():.3f} ({status(table.max().max())})")


if __name__ == "__main__":
    main()
