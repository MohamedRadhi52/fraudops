"""Static report for GitHub Pages, filled with the results published in reports/."""

import json
import shutil
from pathlib import Path
from string import Template

REPORTS = Path("reports")
TEMPLATE = Path("site/index.html")
OUTPUT = Path("_site")
CARD_MODELS = {
    "rules": "Règles métier",
    "logistic": "Régression logistique",
    "lightgbm": "LightGBM",
}


def number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def interval(values: list[float], scale: float = 1, digits: int = 2, unit: str = "") -> str:
    """Value followed by its 95% interval, for a table cell."""
    value, low, high = (number(scale * v, digits) for v in values)
    return f"{value}{unit} <small>[{low} ; {high}]</small>"


def percent(value: float, digits: int = 1) -> str:
    return f"{number(100 * value, digits)} %"


def row(cells: list[str], best: bool = False) -> str:
    attribute = ' class="best"' if best else ""
    return f"<tr{attribute}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def values() -> dict[str, str]:
    """Every number of the page, formatted."""
    models, cost, calibration = load("models.json"), load("cost.json"), load("calibration.json")
    explain, drift, baf = load("explain.json"), load("drift.json"), load("baf/metrics.json")

    def euros(name: str) -> str:
        return interval(cost[name]["cost"], 1 / 1000, 1, " k€")

    models_rows = [
        row(
            [
                label,
                interval(models["models"][name]["card_precision"], 100, 1, " %"),
                interval(models["models"][name]["auc_pr"]),
                euros(name),
            ],
            best=name == "lightgbm",
        )
        for name, label in CARD_MODELS.items()
    ]
    models_rows.append(
        row(["Modèle parfait", percent(models["card_precision_ceiling"]), "1", euros("perfect")])
    )
    models_rows.append(row(["Aucun contrôle", "", "", euros("no_control")]))

    calibration_rows = [
        row(
            [
                label,
                percent(calibration[name]["mean_probability"], 2),
                number(calibration[name]["brier"], 5),
                interval(calibration[name]["expected_loss_rule"]["cost"], 1 / 1000, 1, " k€"),
            ]
        )
        for name, label in [
            ("lightgbm", "LightGBM"),
            ("weighted", "LightGBM pondéré"),
            ("weighted_calibrated", "LightGBM pondéré, calibré"),
        ]
    ]
    baf_rows = [
        row(
            [
                label,
                interval(baf["models"][name]["recall"], 100, 1, " %"),
                interval(baf["models"][name]["fpr"], 100, 1, " %"),
                interval(baf["models"][name]["recall_threshold_on_test"], 100, 1, " %"),
            ],
            best=name == "lightgbm",
        )
        for name, label in [("logistic", "Régression logistique"), ("lightgbm", "LightGBM")]
    ]
    fairness = baf["fairness"]
    fairness_rows = [
        row(
            [
                label,
                percent(fairness[name]["by_group"]["50 ans et plus"]["fpr"]),
                percent(fairness[name]["by_group"]["moins de 50 ans"]["fpr"]),
                interval(fairness[name]["fpr_ratio"]),
                interval(fairness[name]["recall"], 100, 1, " %"),
            ]
        )
        for name, label in [
            ("single_threshold", "Seuil unique"),
            ("group_thresholds", "Un seuil par groupe d'âge"),
            ("without_age", "Seuil unique, modèle sans l'âge"),
        ]
    ]
    stopped = models["lightgbm_share_of_frauds_stopped"]
    example = explain["example"]
    weekly_psi = max(max(week.values()) for week in drift["weekly_psi"].values())
    test_months = [baf["monthly_psi"]["mois 6"], baf["monthly_psi"]["mois 7"]]
    drifting = [
        name
        for name in test_months[0]
        if name != "score" and min(m[name] for m in test_months) >= 0.25
    ]
    return {
        "cost_none_short": f"{number(cost['no_control']['cost'][0] / 1000, 0)} k€",
        "cost_rules_short": f"{number(cost['rules']['cost'][0] / 1000, 0)} k€",
        "cost_logistic_short": f"{number(cost['logistic']['cost'][0] / 1000, 0)} k€",
        "cost_lightgbm_short": f"{number(cost['lightgbm']['cost'][0] / 1000, 0)} k€",
        "cost_perfect_short": f"{number(cost['perfect']['cost'][0] / 1000, 0)} k€",
        "cost_lightgbm": euros("lightgbm"),
        "cost_full": euros("lightgbm_full_capacity"),
        "controls_lightgbm": number(cost["lightgbm"]["controls_per_day"], 0),
        "ceiling": percent(models["card_precision_ceiling"]),
        "models_rows": "".join(models_rows),
        "stopped_unknown": percent(stopped["scenario 2, terminal without known fraud"], 0),
        "stopped_known": percent(stopped["scenario 2"], 0),
        "stopped_s3": percent(stopped["scenario 3"], 0),
        "stopped_s1": percent(stopped["scenario 1"], 0),
        "fraud_rate": percent(calibration["fraud_rate"], 2),
        "calibration_rows": "".join(calibration_rows),
        "alerts": str(explain["alerts"]),
        "alert_fraud_share": percent(explain["fraud_share_of_alerts"], 0),
        "example_amount": f"{number(example['features']['tx_amount'], 2)} €",
        "example_habit": f"{number(example['features']['customer_avg_amount_30d'], 0)} €",
        "example_loss": f"{number(example['expected_loss'], 0)} €",
        "example_reasons": ", ".join(example["reasons"]),
        "baf_applications": f"{sum(baf['applications'].values()):,}".replace(",", " "),
        "baf_rows": "".join(baf_rows),
        "fairness_rows": "".join(fairness_rows),
        "mitigation_loss": interval(fairness["mitigation_recall_loss"], 100, 1, " point"),
        "max_weekly_psi": number(weekly_psi, 3),
        "baf_drifting": str(len(drifting)),
        "baf_score_psi": number(max(month["score"] for month in test_months), 3),
        "baf_test_fpr": percent(baf["models"]["lightgbm"]["fpr"][0]),
    }


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    shutil.copytree(REPORTS / "figures", OUTPUT / "figures")
    shutil.copytree(REPORTS / "baf" / "figures", OUTPUT / "figures" / "baf")
    page = Template(TEMPLATE.read_text()).substitute(values())
    (OUTPUT / "index.html").write_text(page)
    print(f"Rapport écrit dans {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
