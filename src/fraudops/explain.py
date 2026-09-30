"""What the card model looks at: SHAP values of the production LightGBM."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fraudops.cost import CONTROL_COST
from fraudops.evaluate import REPORT_DIR
from fraudops.features import FEATURES_PATH
from fraudops.models import FEATURES, fit_lightgbm
from fraudops.validate import split

MODEL_PATH = Path("models/lightgbm.txt")
DEPLOYMENT_DAY = 183  # the day after the data ends


def production_model(features: pd.DataFrame, params: dict) -> LGBMClassifier:
    """LightGBM trained on the last 28 days whose labels are known, as if deployed today."""
    train, _ = split(features, DEPLOYMENT_DAY)
    return fit_lightgbm(train, params)


def shap_values(model: LGBMClassifier, transactions: pd.DataFrame) -> pd.DataFrame:
    """SHAP value of each feature for each transaction, in log-odds (TreeSHAP of LightGBM)."""
    contributions = model.predict(transactions[FEATURES], pred_contrib=True)
    return pd.DataFrame(contributions[:, :-1], columns=FEATURES, index=transactions.index)


def top_reasons(shap_row: pd.Series, n: int = 3) -> dict[str, float]:
    """The n features that push the score up the most."""
    return shap_row.nlargest(n).to_dict()


def plot_importance(importance: pd.DataFrame, path) -> None:
    """Mean absolute SHAP value of each feature, on all transactions and on the alerts."""
    importance = importance.sort_values("alerts")
    y = np.arange(len(importance))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y + 0.2, importance["alerts"], 0.4, label="Transactions signalées", color="#1F3F7A")
    ax.barh(y - 0.2, importance["all"], 0.4, label="Toutes les transactions", color="#9DB3D9")
    ax.set_yticks(y, importance.index)
    ax.set_xlabel("Contribution moyenne au score, en valeur absolue (log-odds)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    params = json.loads((REPORT_DIR / "models.json").read_text())["lightgbm_params"]
    model = production_model(features, params)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.booster_.save_model(MODEL_PATH)

    # the week after the training window, never seen by the model
    week = features[features["tx_time_days"] >= DEPLOYMENT_DAY - 7]
    shap = shap_values(model, week)
    expected_loss = model.predict_proba(week[FEATURES])[:, 1] * week["tx_amount"]
    alerts = expected_loss >= CONTROL_COST
    importance = pd.DataFrame({"all": shap.abs().mean(), "alerts": shap[alerts].abs().mean()})
    example = expected_loss.where(week["tx_fraud"] == 1).idxmax()
    report = {
        "alerts": int(alerts.sum()),
        "fraud_share_of_alerts": week.loc[alerts, "tx_fraud"].mean(),
        "importance": importance.sort_values("alerts", ascending=False).to_dict(orient="index"),
        "example": {
            "features": week.loc[example, FEATURES].to_dict(),
            "expected_loss": expected_loss[example],
            "reasons": top_reasons(shap.loc[example]),
        },
    }
    (REPORT_DIR / "explain.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_importance(importance, REPORT_DIR / "figures" / "shap.png")
    print(json.dumps(report["example"], indent=2))


if __name__ == "__main__":
    main()
