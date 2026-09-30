import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from fraudops.calibration import calibrate


def inflated_scores(n, seed):
    """Scores ranking well but ten times too high on average, like a weighted model."""
    rng = np.random.default_rng(seed)
    probability = rng.uniform(0, 0.02, n)
    fraud = (rng.random(n) < probability).astype(int)
    return pd.DataFrame({"score": np.sqrt(probability / 2), "tx_fraud": fraud})


def test_calibration_brings_probabilities_back_to_the_fraud_rate():
    validation, test = inflated_scores(50_000, seed=0), inflated_scores(50_000, seed=1)
    calibrated = calibrate(validation, test, "score")
    assert test["score"].mean() > 5 * test["tx_fraud"].mean()
    assert abs(calibrated.mean() - test["tx_fraud"].mean()) < 0.002
    assert brier_score_loss(test["tx_fraud"], calibrated) < brier_score_loss(
        test["tx_fraud"], test["score"]
    )
