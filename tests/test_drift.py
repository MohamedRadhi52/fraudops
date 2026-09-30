import pandas as pd

from fraudops.drift import status, summary_table


def test_status_follows_the_alert_levels():
    assert [status(value) for value in (0.02, 0.15, 0.4)] == [
        "stable",
        "surveiller",
        "ré-entraîner",
    ]


def test_summary_counts_the_variables_past_each_level():
    table = pd.DataFrame(
        {"mois 6": [0.05, 0.15, 0.30, 0.12]},
        index=["income", "velocity_6h", "velocity_4w", "score"],
    )
    assert "| mois 6 | 0,120 | 1 | 1 |" in summary_table(table, "score")
