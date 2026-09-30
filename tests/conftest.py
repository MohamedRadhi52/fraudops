import numpy as np
import pandas as pd
import pytest

from fraudops.features import build_features
from fraudops.simulate import generate_dataset


@pytest.fixture(scope="session")
def small_transactions():
    """About 25,000 transactions over 60 days, enough to exercise every scenario quickly."""
    return generate_dataset(n_customers=200, n_terminals=1_000, nb_days=60, radius=8, seed=1)


@pytest.fixture(scope="session")
def small_features(small_transactions):
    return build_features(small_transactions)


def fake_baf(n: int, seed: int) -> pd.DataFrame:
    """Random applications with the 32 columns and value ranges of the BAF datasheet.

    Fraud depends on the risk score, the email and the age, so that models find a signal and
    older applicants get more false positives, as in the real data.
    """
    rng = np.random.default_rng(seed)
    age = rng.choice(
        np.arange(10, 100, 10), n, p=[0.02, 0.25, 0.3, 0.2, 0.12, 0.07, 0.03, 0.007, 0.003]
    )
    credit_risk_score = rng.integers(-191, 390, n)
    email_is_free = rng.integers(0, 2, n)
    logit = -4.5 + credit_risk_score / 150 + email_is_free + (age >= 50)
    return pd.DataFrame(
        {
            "fraud_bool": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
            "income": rng.choice(np.arange(1, 10) / 10, n),
            "name_email_similarity": rng.random(n),
            "prev_address_months_count": np.where(rng.random(n) < 0.7, -1, rng.integers(0, 381, n)),
            "current_address_months_count": rng.integers(-1, 430, n),
            "customer_age": age,
            "days_since_request": rng.exponential(1, n),
            "intended_balcon_amount": rng.uniform(-16, 114, n),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], n),
            "zip_count_4w": rng.integers(1, 6831, n),
            "velocity_6h": rng.uniform(-175, 16818, n),
            "velocity_24h": rng.uniform(1297, 9586, n),
            "velocity_4w": rng.uniform(2825, 7020, n),
            "bank_branch_count_8w": rng.integers(0, 2405, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 40, n),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], n),
            "credit_risk_score": credit_risk_score,
            "email_is_free": email_is_free,
            "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], n),
            "phone_home_valid": rng.integers(0, 2, n),
            "phone_mobile_valid": rng.integers(0, 2, n),
            "bank_months_count": rng.integers(-1, 33, n),
            "has_other_cards": rng.integers(0, 2, n),
            "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], n),
            "foreign_request": rng.integers(0, 2, n),
            "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[0.99, 0.01]),
            "session_length_in_minutes": rng.uniform(-1, 107, n),
            "device_os": rng.choice(["windows", "macintosh", "linux", "x11", "other"], n),
            "keep_alive_session": rng.integers(0, 2, n),
            "device_distinct_emails_8w": rng.integers(-1, 3, n),
            "device_fraud_count": np.zeros(n, dtype=int),
            "month": rng.integers(0, 8, n),
        }
    )


@pytest.fixture(scope="session")
def fake_baf_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("baf") / "Base.csv"
    fake_baf(20_000, seed=0).to_csv(path, index=False)
    return path
