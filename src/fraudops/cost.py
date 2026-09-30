"""Alert threshold chosen by cost, within the daily investigation capacity."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudops import metrics
from fraudops.evaluate import LABELS, REPORT_DIR, SCORES_PATH

CONTROL_COST = 10  # euros per investigated card, see docs/cadrage.md


def daily_cost(scored: pd.DataFrame, score: str, threshold: float) -> pd.DataFrame:
    """Controls and cost of each day when cards scoring at least `threshold` are investigated.

    At most 100 cards are investigated a day. Each investigation costs CONTROL_COST euros and
    each fraud that is not stopped costs its amount.
    """
    investigated = metrics.investigations(scored, score, threshold=threshold)
    frauds = scored[scored["tx_fraud"] == 1]
    missed = frauds[~metrics.stopped(frauds, investigated)]
    daily = pd.DataFrame(
        {
            "controls": investigated.groupby(level="tx_time_days").size(),
            "missed": missed.groupby("tx_time_days")["tx_amount"].sum(),
        }
    )
    # Days without any control or any missed fraud count for zero, not for a missing value.
    daily = daily.reindex(np.sort(scored["tx_time_days"].unique())).fillna(0)
    daily["cost"] = CONTROL_COST * daily["controls"] + daily["missed"]
    return daily


def summarize_cost(daily: pd.DataFrame, threshold: float | None = None) -> dict:
    """Mean controls per day and total cost as [value, low, high], bootstrapping the days."""
    cost = daily["cost"]
    interval = metrics.bootstrap_interval(lambda days: cost[days].sum(), cost.index.to_numpy())
    return {
        "threshold": threshold,
        "controls_per_day": daily["controls"].mean(),
        "cost": [cost.sum(), *interval],
    }


def cost_curve(scored: pd.DataFrame, score: str) -> pd.DataFrame:
    """Total cost for thresholds going from a full capacity every day to almost no alert."""
    levels = 1 - np.geomspace(0.05, 1e-5, 60)
    rows = []
    for threshold in np.unique(np.quantile(scored[score], levels)):
        daily = daily_cost(scored, score, threshold)
        rows.append((threshold, daily["controls"].mean(), daily["cost"].sum()))
    return pd.DataFrame(rows, columns=["threshold", "controls_per_day", "cost"])


def plot_cost_curves(curves: dict[str, pd.DataFrame], report: dict, path) -> None:
    """Test cost of each model against the number of controls, with the chosen thresholds."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, curve in curves.items():
        (line,) = ax.plot(curve["controls_per_day"], curve["cost"] / 1000, label=LABELS[name])
        point = report[name]
        ax.plot(point["controls_per_day"], point["cost"][0] / 1000, "o", color=line.get_color())
    perfect = report["perfect"]["cost"][0] / 1000
    ax.axhline(perfect, color="#C44E52", linestyle="--")
    ax.text(100, perfect, "modèle parfait", color="#C44E52", ha="right", va="bottom")
    ax.set_xlim(0, metrics.CAPACITY)
    ax.set_ylim(0, None)
    ax.set_xlabel("Cartes contrôlées par jour, en moyenne (points : seuils choisis en validation)")
    ax.set_ylabel("Coût sur les 8 semaines de test (k€)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strategies(report: dict, path) -> None:
    """Total cost of each strategy, with its confidence interval."""
    names = ["no_control", *LABELS, "perfect"]
    labels = ["Aucun contrôle", *LABELS.values(), "Modèle parfait"]
    value, low, high = np.array([report[name]["cost"] for name in names]).T / 1000
    colors = ["#999999", "#4C72B0", "#4C72B0", "#1F3F7A", "#999999"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(labels, value, xerr=[value - low, high - value], color=colors, capsize=4)
    for y, v in enumerate(value):
        ax.text(v + (high[y] - v) + 4, y, f"{v:.0f} k€", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15 * high.max())
    ax.set_xlabel(
        "Fraudes non stoppées et contrôles à 10 €, sur les 8 semaines de test (k€, IC 95 %)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    scores = pd.read_parquet(SCORES_PATH)
    validation = scores[scores["period"] == "validation"]
    test = scores[scores["period"] == "test"]

    report, curves = {"control_cost": CONTROL_COST}, {}
    for name in LABELS:
        validation_curve = cost_curve(validation, name)
        threshold = validation_curve.loc[validation_curve["cost"].idxmin(), "threshold"]
        report[name] = summarize_cost(daily_cost(test, name, threshold), threshold)
        curves[name] = cost_curve(test, name)
    report["lightgbm_full_capacity"] = summarize_cost(daily_cost(test, "lightgbm", -np.inf))
    report["no_control"] = summarize_cost(daily_cost(test, "lightgbm", np.inf))
    perfect = test.assign(perfect=test["tx_fraud"])
    report["perfect"] = summarize_cost(daily_cost(perfect, "perfect", 1))

    (REPORT_DIR / "cost.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_cost_curves(curves, report, REPORT_DIR / "figures" / "cost.png")
    plot_strategies(report, REPORT_DIR / "figures" / "cost_by_strategy.png")
    for name in [*LABELS, "lightgbm_full_capacity", "no_control", "perfect"]:
        value, low, high = (np.array(report[name]["cost"]) / 1000).round(1)
        controls = report[name]["controls_per_day"]
        print(f"{name} : {value} k€ [{low} ; {high}], {controls:.0f} contrôles par jour")


if __name__ == "__main__":
    main()
