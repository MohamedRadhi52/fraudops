import pytest

from fraudops.simulate import generate_dataset


@pytest.fixture(scope="session")
def small_transactions():
    """About 25,000 transactions over 60 days, enough to exercise every scenario quickly."""
    return generate_dataset(n_customers=200, n_terminals=1_000, nb_days=60, radius=8, seed=1)
