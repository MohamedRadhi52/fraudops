from fraudops.evaluate import MODELS, prequential_scores


def test_prequential_scores_cover_every_test_week(small_features):
    scores = prequential_scores(small_features, weeks=(45, 52))
    assert scores["tx_time_days"].min() == 45
    assert scores["tx_time_days"].max() == 58
    assert scores[list(MODELS)].notna().all().all()
