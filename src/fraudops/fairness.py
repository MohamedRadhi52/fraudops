"""Fairness across ages on BAF: false positive rates by group, before and after mitigation.

The BAF paper measures predictive equality: the FPR of the group flagged least often divided by
the FPR of the group flagged most often, between applicants of 50 and over and the others.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, false_positive_rate, true_positive_rate

from fraudops.metrics import bootstrap_interval, threshold_at_fpr

OLDER = "50 ans et plus"
YOUNGER = "moins de 50 ans"
LABELS = {
    "single_threshold": "Seuil unique",
    "group_thresholds": "Un seuil par groupe d'âge",
    "without_age": "Seuil unique, modèle sans l'âge",
}


def age_group(ages: pd.Series) -> np.ndarray:
    return np.where(ages >= 50, OLDER, YOUNGER)


def group_thresholds(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict:
    """One threshold per group, each flagging 5% of the group's legitimate applications."""
    return {
        group: threshold_at_fpr(labels[groups == group], scores[groups == group])
        for group in (OLDER, YOUNGER)
    }


def audit(labels: np.ndarray, flagged: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """False positive rate, recall and size of each group, computed with Fairlearn."""
    frame = MetricFrame(
        metrics={"fpr": false_positive_rate, "recall": true_positive_rate, "applications": count},
        y_true=labels,
        y_pred=flagged,
        sensitive_features=groups,
    )
    return frame.by_group


def fpr_ratio(labels: np.ndarray, flagged: np.ndarray, groups: np.ndarray) -> float:
    """Lowest group FPR divided by the highest one (1 means equal false positive rates)."""
    legitimate = labels == 0
    fprs = [flagged[legitimate & (groups == group)].mean() for group in (OLDER, YOUNGER)]
    return min(fprs) / max(fprs)


def summarize_variant(
    labels: np.ndarray, flagged: np.ndarray, groups: np.ndarray, ages: np.ndarray
) -> dict:
    """FPR ratio and recall as [value, low, high], and FPR by group and by age decade."""
    rows = np.arange(len(labels))
    frauds = np.flatnonzero(labels == 1)
    return {
        "fpr_ratio": [
            fpr_ratio(labels, flagged, groups),
            *bootstrap_interval(lambda s: fpr_ratio(labels[s], flagged[s], groups[s]), rows),
        ],
        "recall": [
            flagged[frauds].mean(),
            *bootstrap_interval(lambda s: flagged[s].mean(), frauds),
        ],
        "by_group": audit(labels, flagged, groups).to_dict(orient="index"),
        "fpr_by_age": audit(labels, flagged, ages)["fpr"].to_dict(),
    }


def fairness_report(validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Compare a single threshold, one threshold per age group and a model trained without age.

    Both frames hold the columns label, age, score and score_without_age.
    """
    groups, ages, labels = age_group(test["age"]), test["age"].to_numpy(), test["label"].to_numpy()
    validation_labels = validation["label"].to_numpy()
    thresholds = group_thresholds(
        validation_labels, validation["score"].to_numpy(), age_group(validation["age"])
    )
    single = threshold_at_fpr(validation_labels, validation["score"].to_numpy())
    without_age = threshold_at_fpr(validation_labels, validation["score_without_age"].to_numpy())
    flagged = {
        "single_threshold": test["score"].to_numpy() > single,
        "group_thresholds": test["score"].to_numpy() > pd.Series(groups).map(thresholds).to_numpy(),
        "without_age": test["score_without_age"].to_numpy() > without_age,
    }
    report = {
        name: summarize_variant(labels, flags, groups, ages) for name, flags in flagged.items()
    }
    # Paired bootstrap: recall lost by the mitigation, on the same resampled frauds.
    frauds = labels == 1
    loss = flagged["single_threshold"][frauds].astype(float) - flagged["group_thresholds"][frauds]
    rows = np.arange(len(loss))
    report["mitigation_recall_loss"] = [
        loss.mean(),
        *bootstrap_interval(lambda s: loss[s].mean(), rows),
    ]
    report["group_thresholds_values"] = thresholds | {"single": single}
    return report


def results_table(report: dict) -> str:
    """Markdown table of the fairness results."""
    lines = [
        f"| Variante | FPR, {OLDER} | FPR, {YOUNGER} | Ratio de FPR | Rappel |",
        "|---|---|---|---|---|",
    ]
    for name, label in LABELS.items():
        result = report[name]
        older, younger = (100 * result["by_group"][group]["fpr"] for group in (OLDER, YOUNGER))
        ratio = "{:.2f} [{:.2f} ; {:.2f}]".format(*result["fpr_ratio"])
        recall = "{:.1f} % [{:.1f} ; {:.1f}]".format(*(100 * np.array(result["recall"])))
        lines.append(
            f"| {label} | {older:.1f} % | {younger:.1f} % | {ratio} | {recall} |".replace(".", ",")
        )
    return "\n".join(lines) + "\n"


def plot_fpr_by_age(report: dict, path) -> None:
    """False positive rate of each age decade, with a single threshold and one per group."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.4
    for offset, name in [(-width / 2, "single_threshold"), (width / 2, "group_thresholds")]:
        fpr = pd.Series(report[name]["fpr_by_age"]).sort_index()
        ax.bar(np.arange(len(fpr)) + offset, 100 * fpr.to_numpy(), width, label=LABELS[name])
    ax.set_xticks(np.arange(len(fpr)), [f"{age} ans" for age in fpr.index])
    ax.axhline(5, color="grey", linestyle=":")
    ax.text(len(fpr) - 0.5, 5, "objectif : 5 %", color="grey", ha="right", va="bottom")
    ax.set_ylabel("Demandes légitimes signalées (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(report: dict, path) -> None:
    """Recall against FPR ratio for each variant: the performance and fairness trade-off."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, label in LABELS.items():
        recall, recall_low, recall_high = 100 * np.array(report[name]["recall"])
        ratio, ratio_low, ratio_high = report[name]["fpr_ratio"]
        ax.errorbar(
            recall,
            ratio,
            xerr=[[recall - recall_low], [recall_high - recall]],
            yerr=[[ratio - ratio_low], [ratio_high - ratio]],
            fmt="o",
            capsize=4,
            label=label,
        )
    ax.axhline(0.3, color="grey", linestyle=":")
    ax.text(ax.get_xlim()[0], 0.31, " meilleurs modèles du papier BAF", color="grey")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rappel à 5 % de faux positifs (%)")
    ax.set_ylabel("Ratio de FPR entre groupes d'âge (1 = égalité)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
