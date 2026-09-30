import numpy as np
import pandas as pd

from fraudops.explain import shap_values, top_reasons
from fraudops.models import FEATURES, LIGHTGBM_GRID, fit_lightgbm
from fraudops.validate import split


def test_shap_values_add_up_to_the_score(small_features):
    train, test = split(small_features, test_start=45)
    model = fit_lightgbm(train, LIGHTGBM_GRID[0])
    bias = model.predict(test[FEATURES], pred_contrib=True)[:, -1]
    log_odds = shap_values(model, test).sum(axis=1) + bias
    probability = model.predict_proba(test[FEATURES])[:, 1]
    assert np.allclose(1 / (1 + np.exp(-log_odds)), probability)


def test_top_reasons_are_the_largest_contributions():
    shap_row = pd.Series({"tx_amount": 2.0, "terminal_risk_7d": 3.0, "a": -1.0, "b": 0.5})
    assert list(top_reasons(shap_row)) == ["terminal_risk_7d", "tx_amount", "b"]
