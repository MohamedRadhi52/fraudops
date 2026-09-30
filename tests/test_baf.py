import numpy as np
import pytest

from fraudops.baf import CATEGORICAL, load, run, split
from fraudops.metrics import TARGET_FPR, rates, threshold_at_fpr


@pytest.fixture(scope="module")
def baf(fake_baf_csv):
    return load(fake_baf_csv)


@pytest.fixture(scope="module")
def result(baf):
    return run(baf)[0]


def test_categories_are_typed_and_months_split(baf):
    assert all(baf[column].dtype == "category" for column in CATEGORICAL)
    train, validation, test = split(baf)
    assert train["month"].max() == 4
    assert set(validation["month"]) == {5}
    assert set(test["month"]) == {6, 7}


def test_threshold_flags_five_percent_of_legitimate_applications():
    rng = np.random.default_rng(0)
    labels = (rng.random(100_000) < 0.01).astype(int)
    scores = rng.random(100_000) + labels
    threshold = threshold_at_fpr(labels, scores)
    assert rates(labels, scores, threshold)["fpr"] == pytest.approx(TARGET_FPR, abs=0.001)


def test_models_find_frauds_on_a_fake_sample(result):
    for model in result["models"].values():
        low, value, high = model["recall"][1], model["recall"][0], model["recall"][2]
        assert low <= value <= high
        # Random scores would flag about 5% of the frauds.
        assert value > 3 * TARGET_FPR
    assert len(result["lightgbm_recall_gain"]) == 3
