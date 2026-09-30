import numpy as np
import pytest
from fairlearn.metrics import MetricFrame, false_positive_rate

from fraudops.fairness import OLDER, YOUNGER, fpr_ratio, group_thresholds
from fraudops.metrics import TARGET_FPR


@pytest.fixture
def applications():
    """Older applicants get higher scores, whatever their label."""
    rng = np.random.default_rng(0)
    groups = np.where(rng.random(50_000) < 0.3, OLDER, YOUNGER)
    labels = (rng.random(50_000) < 0.02).astype(int)
    scores = rng.random(50_000) + labels + 0.3 * (groups == OLDER)
    return labels, scores, groups


def test_fpr_ratio_matches_fairlearn(applications):
    labels, scores, groups = applications
    flagged = scores > 1
    fairlearn = MetricFrame(
        metrics=false_positive_rate, y_true=labels, y_pred=flagged, sensitive_features=groups
    ).ratio()
    assert fpr_ratio(labels, flagged, groups) == pytest.approx(fairlearn)


def test_group_thresholds_flag_five_percent_of_each_group(applications):
    labels, scores, groups = applications
    thresholds = group_thresholds(labels, scores, groups)
    for group in (OLDER, YOUNGER):
        legitimate = (groups == group) & (labels == 0)
        assert (scores[legitimate] > thresholds[group]).mean() == pytest.approx(
            TARGET_FPR, abs=0.002
        )


def test_one_threshold_per_group_equalizes_false_positives(baf_report):
    fairness = baf_report["fairness"]
    assert (
        fairness["group_thresholds"]["fpr_ratio"][0] > fairness["single_threshold"]["fpr_ratio"][0]
    )
    assert set(fairness["single_threshold"]["by_group"]) == {OLDER, YOUNGER}
