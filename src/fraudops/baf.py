"""Bank account opening fraud (BAF suite, Feedzai, NeurIPS 2022): recall at 5% FPR.

The data is under a non-commercial licence: the GitHub Actions workflow downloads it, and only
metrics and figures are published, never the data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudops.metrics import bootstrap_interval, format_interval, rates, threshold_at_fpr

BAF_PATH = Path("data/baf/Base.csv")
MODEL_PATH = Path("data/baf/lightgbm.txt")
REPORT_DIR = Path("reports/baf")
LABEL = "fraud_bool"
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
LABELS = {"logistic": "Régression logistique", "lightgbm": "LightGBM"}


def load(path: Path = BAF_PATH) -> pd.DataFrame:
    """BAF applications, with categorical columns typed as such."""
    return pd.read_csv(path).astype({column: "category" for column in CATEGORICAL})


def split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on months 0 to 4, validate on month 5, test on months 6 and 7."""
    month = data["month"]
    return data[month <= 4], data[month == 5], data[month >= 6]


def inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: every column but the label and the month, which only serves to split."""
    return data.drop(columns=[LABEL, "month"])


def lightgbm(train: pd.DataFrame, validation: pd.DataFrame) -> LGBMClassifier:
    """LightGBM stopped early on the validation month."""
    model = LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, deterministic=True, force_row_wise=True, verbose=-1
    )
    return model.fit(
        inputs(train),
        train[LABEL],
        eval_X=inputs(validation),
        eval_y=validation[LABEL],
        eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False)],
    )


def logistic_regression(train: pd.DataFrame) -> Pipeline:
    """Logistic regression on standardized numbers and one-hot categories."""
    model = make_pipeline(
        make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_exclude="category")),
            (
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include="category"),
            ),
        ),
        LogisticRegression(max_iter=1000),
    )
    return model.fit(inputs(train), train[LABEL])


def summarize_model(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Recall and FPR on the test months at `threshold`, and recall at 5% FPR measured there."""
    rows = np.arange(len(labels))

    def at_threshold(sample: np.ndarray) -> dict[str, float]:
        return rates(labels[sample], scores[sample], threshold)

    def at_test_fpr(sample: np.ndarray) -> float:
        test_threshold = threshold_at_fpr(labels[sample], scores[sample])
        return rates(labels[sample], scores[sample], test_threshold)["recall"]

    return {
        "recall": [
            at_threshold(rows)["recall"],
            *bootstrap_interval(lambda sample: at_threshold(sample)["recall"], rows),
        ],
        "fpr": [
            at_threshold(rows)["fpr"],
            *bootstrap_interval(lambda sample: at_threshold(sample)["fpr"], rows),
        ],
        "recall_threshold_on_test": [at_test_fpr(rows), *bootstrap_interval(at_test_fpr, rows)],
    }


def evaluate(validation: pd.DataFrame, test: pd.DataFrame, scores: dict) -> dict:
    """Recall at 5% FPR of each model, with 95% intervals from resampling the applications.

    The threshold is set on the validation month, then kept on the test months. The recall at 5%
    FPR measured on the test months themselves, as in the BAF paper, is given for comparison.
    """
    labels = test[LABEL].to_numpy()
    models, flagged = {}, {}
    for name, (validation_scores, test_scores) in scores.items():
        threshold = threshold_at_fpr(validation[LABEL].to_numpy(), validation_scores)
        models[name] = summarize_model(labels, test_scores, threshold)
        flagged[name] = test_scores > threshold
    # Paired bootstrap: both models are compared on the same resampled frauds.
    gain = flagged["lightgbm"][labels == 1].astype(float) - flagged["logistic"][labels == 1]
    rows = np.arange(len(gain))
    return {
        "models": models,
        "lightgbm_recall_gain": [gain.mean(), *bootstrap_interval(lambda s: gain[s].mean(), rows)],
    }


def results_table(report: dict) -> str:
    """Markdown table of the results, published with the metrics."""
    lines = [
        "| Modèle | Rappel, seuil fixé en validation | FPR obtenu sur le test "
        "| Rappel à 5 % de FPR mesuré sur le test |",
        "|---|---|---|---|",
    ]
    for name, result in report["models"].items():
        recall, fpr, paper = (
            format_interval(*result[metric], percent=True)
            for metric in ("recall", "fpr", "recall_threshold_on_test")
        )
        lines.append(f"| {LABELS[name]} | {recall} | {fpr} | {paper} |")
    return "\n".join(lines) + "\n"


def run(data: pd.DataFrame) -> tuple[dict, LGBMClassifier]:
    """Train both models and evaluate them on the test months."""
    train, validation, test = split(data)
    models = {"logistic": logistic_regression(train), "lightgbm": lightgbm(train, validation)}
    scores = {
        name: tuple(model.predict_proba(inputs(part))[:, 1] for part in (validation, test))
        for name, model in models.items()
    }
    report = {
        "applications": {"train": len(train), "validation": len(validation), "test": len(test)},
        "fraud_rate": {
            "train": train[LABEL].mean(),
            "validation": validation[LABEL].mean(),
            "test": test[LABEL].mean(),
        },
    } | evaluate(validation, test, scores)
    return report, models["lightgbm"]


def main() -> None:
    report, model = run(load())
    model.booster_.save_model(MODEL_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    table = results_table(report)
    (REPORT_DIR / "RESULTS.md").write_text(
        "# Résultats sur BAF\n\nGénéré par le workflow `baf.yml` sur la variante Base, "
        "qui n'est jamais publiée.\n\n" + table
    )
    print(table)


if __name__ == "__main__":
    main()
