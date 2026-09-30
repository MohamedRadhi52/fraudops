"""Calibration of the LightGBM probabilities, and what it changes for the cost decision."""

import json
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from fraudops import models
from fraudops.cost import CONTROL_COST, daily_cost, summarize_cost
from fraudops.evaluate import REPORT_DIR, SCORES_PATH, prequential_scores
from fraudops.features import FEATURES_PATH

LABELS = {
    "lightgbm": "LightGBM",
    "lightgbm_calibrated": "LightGBM calibré",
    "weighted": "LightGBM pondéré",
    "weighted_calibrated": "LightGBM pondéré, calibré",
}


def calibrate(validation: pd.DataFrame, test: pd.DataFrame, score: str) -> pd.Series:
    """Test probabilities corrected by an isotonic regression fitted on the validation weeks."""
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        validation[score], validation["tx_fraud"]
    )
    return pd.Series(isotonic.predict(test[score]), index=test.index)


def weighted_lightgbm_scores() -> pd.Series:
    """Scores of the tuned LightGBM trained with balanced class weights, a common practice."""
    params = json.loads((REPORT_DIR / "models.json").read_text())["lightgbm_params"]
    fit_predict = partial(models.lightgbm, params=params | {"class_weight": "balanced"})
    scores = prequential_scores(pd.read_parquet(FEATURES_PATH), {"weighted": fit_predict})
    return scores.set_index("transaction_id")["weighted"]


def plot_reliability(test: pd.DataFrame, path) -> None:
    """Observed fraud rate against predicted probability, by tenths of probability."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="calibration parfaite")
    for name, style in [("weighted", "o--"), ("weighted_calibrated", "o-"), ("lightgbm", "s-")]:
        observed, predicted = calibration_curve(test["tx_fraud"], test[name], n_bins=10)
        ax.plot(predicted, observed, style, label=LABELS[name])
    ax.set_xlabel("Probabilité de fraude prédite")
    ax.set_ylabel("Part de fraudes observée (semaines de test)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    scores = pd.read_parquet(SCORES_PATH)
    scores["weighted"] = scores["transaction_id"].map(weighted_lightgbm_scores())
    validation = scores[scores["period"] == "validation"]
    test = scores[scores["period"] == "test"].copy()
    for name in ("lightgbm", "weighted"):
        test[f"{name}_calibrated"] = calibrate(validation, test, name)

    report = {"fraud_rate": test["tx_fraud"].mean()}
    for name in LABELS:
        # Expected loss rule: investigate a card when probability x amount exceeds the control cost.
        expected_loss = test.assign(expected_loss=test[name] * test["tx_amount"])
        report[name] = {
            "brier": brier_score_loss(test["tx_fraud"], test[name]),
            "mean_probability": test[name].mean(),
            "expected_loss_rule": summarize_cost(
                daily_cost(expected_loss, "expected_loss", CONTROL_COST)
            ),
        }
    (REPORT_DIR / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_reliability(test, REPORT_DIR / "figures" / "calibration.png")

    print(f"Part de fraudes dans les semaines de test : {report['fraud_rate']:.2%}")
    for name, label in LABELS.items():
        result, rule = report[name], report[name]["expected_loss_rule"]
        print(
            f"{label} : Brier {result['brier']:.5f}, "
            f"probabilité moyenne {result['mean_probability']:.2%}, "
            f"règle de perte attendue {rule['cost'][0] / 1000:.1f} k€, "
            f"{rule['controls_per_day']:.0f} contrôles par jour"
        )


if __name__ == "__main__":
    main()
