import json

import numpy as np
import pytest

import fraudops.baf as baf
from fraudops.baf import CATEGORICAL, load, split
from fraudops.metrics import TARGET_FPR, rates, threshold_at_fpr


@pytest.fixture(scope="module")
def applications(fake_baf_csv):
    return load(fake_baf_csv)


def test_categories_are_typed_and_months_split(applications):
    assert all(applications[column].dtype == "category" for column in CATEGORICAL)
    train, validation, test = split(applications)
    assert train["month"].max() == 4
    assert set(validation["month"]) == {5}
    assert set(test["month"]) == {6, 7}


def test_threshold_flags_five_percent_of_legitimate_applications():
    rng = np.random.default_rng(0)
    labels = (rng.random(100_000) < 0.01).astype(int)
    scores = rng.random(100_000) + labels
    threshold = threshold_at_fpr(labels, scores)
    assert rates(labels, scores, threshold)["fpr"] == pytest.approx(TARGET_FPR, abs=0.001)


def test_models_find_frauds_on_a_fake_sample(baf_report):
    for model in baf_report["models"].values():
        low, value, high = model["recall"][1], model["recall"][0], model["recall"][2]
        assert low <= value <= high
        # Random scores would flag about 5% of the frauds.
        assert value > 3 * TARGET_FPR
    assert len(baf_report["lightgbm_recall_gain"]) == 3


def test_main_publishes_metrics_figures_and_results(fake_baf_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(baf, "load", lambda: load(fake_baf_csv))
    monkeypatch.setattr(baf, "MODEL_PATH", tmp_path / "lightgbm.txt")
    monkeypatch.setattr(baf, "REPORT_DIR", tmp_path / "reports")
    baf.main()
    report = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert {"models", "lightgbm_recall_gain", "fairness"} <= set(report)
    assert "Ratio de FPR" in (tmp_path / "reports" / "RESULTS.md").read_text()
    for figure in ("fpr_by_age.png", "tradeoff.png"):
        assert (tmp_path / "reports" / "figures" / figure).exists()
    assert (tmp_path / "lightgbm.txt").exists()
