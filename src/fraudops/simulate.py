"""Card transaction simulator described in the Fraud Detection Handbook (Le Borgne et al., 2022).

Independent implementation of the generator of chapter 3: customers and terminals placed on a
100 x 100 grid, daily Poisson transactions, and the book's three fraud scenarios.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRANSACTIONS_PATH = Path("data/transactions.parquet")
SECONDS_PER_DAY = 86_400


def customer_profiles(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """Position, spending habits and daily activity of each customer."""
    x = rng.uniform(0, 100, n_customers)
    y = rng.uniform(0, 100, n_customers)
    mean_amount = rng.uniform(5, 100, n_customers)
    mean_nb_tx_per_day = rng.uniform(0, 4, n_customers)
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "mean_amount": mean_amount,
            "std_amount": mean_amount / 2,
            "mean_nb_tx_per_day": mean_nb_tx_per_day,
        }
    )


def terminal_profiles(n_terminals: int, rng: np.random.Generator) -> pd.DataFrame:
    """Position of each terminal."""
    x = rng.uniform(0, 100, n_terminals)
    y = rng.uniform(0, 100, n_terminals)
    return pd.DataFrame({"x": x, "y": y})


def nearby_terminals(
    customers: pd.DataFrame, terminals: pd.DataFrame, radius: float
) -> list[np.ndarray]:
    """Ids of the terminals closer than `radius` to each customer."""
    terminal_xy = terminals[["x", "y"]].to_numpy()
    return [
        np.flatnonzero(np.linalg.norm(terminal_xy - xy, axis=1) < radius)
        for xy in customers[["x", "y"]].to_numpy()
    ]


def genuine_transactions(
    customers: pd.DataFrame,
    terminals: pd.DataFrame,
    nb_days: int,
    start_date: str,
    radius: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Transactions before fraud injection.

    Each day, a customer makes a Poisson number of transactions, around noon, with a Gaussian
    amount, on a terminal drawn uniformly among those within `radius`.
    """
    mean_nb_tx = customers["mean_nb_tx_per_day"].to_numpy()
    counts = rng.poisson(mean_nb_tx[:, None], (len(customers), nb_days))
    # One row per transaction: each (customer, day) pair is repeated as many times as its count.
    customer_id, day = np.indices(counts.shape).reshape(2, -1).repeat(counts.ravel(), axis=1)

    seconds = rng.normal(SECONDS_PER_DAY / 2, 20_000, len(day)).astype(np.int64)
    nearby = nearby_terminals(customers, terminals, radius)
    n_nearby = np.array([len(ids) for ids in nearby])
    kept = (seconds > 0) & (seconds < SECONDS_PER_DAY) & (n_nearby[customer_id] > 0)
    customer_id, day, seconds = customer_id[kept], day[kept], seconds[kept]

    mean = customers["mean_amount"].to_numpy()[customer_id]
    amount = rng.normal(mean, customers["std_amount"].to_numpy()[customer_id])
    negative = amount < 0
    amount[negative] = rng.uniform(0, 2 * mean[negative])

    # All nearby terminals in one flat array: a customer's terminals start at offsets[customer].
    offsets = np.cumsum(n_nearby) - n_nearby
    terminal_id = np.concatenate(nearby)[offsets[customer_id] + rng.integers(n_nearby[customer_id])]

    tx = pd.DataFrame(
        {
            "tx_time_seconds": day * SECONDS_PER_DAY + seconds,
            "tx_time_days": day,
            "customer_id": customer_id,
            "terminal_id": terminal_id,
            "tx_amount": amount.round(2),
        }
    ).sort_values("tx_time_seconds", kind="stable", ignore_index=True)
    tx.insert(0, "transaction_id", np.arange(len(tx)))
    # Microseconds: Spark cannot read the nanosecond timestamps that pandas uses by default.
    datetime = pd.Timestamp(start_date) + pd.to_timedelta(tx["tx_time_seconds"], "s")
    tx.insert(1, "tx_datetime", datetime.astype("datetime64[us]"))
    return tx


def compromised(
    n_entities: int, nb_days: int, per_day: int, duration: int, rng: np.random.Generator
) -> np.ndarray:
    """Which entity is compromised on which day, as an (entity, day) boolean matrix.

    Each day, `per_day` entities are compromised for the next `duration` days.
    """
    matrix = np.zeros((n_entities, nb_days), dtype=bool)
    for day in range(nb_days):
        matrix[rng.choice(n_entities, per_day, replace=False), day : day + duration] = True
    return matrix


def add_frauds(
    tx: pd.DataFrame, n_customers: int, n_terminals: int, nb_days: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Label frauds with the three scenarios of the book.

    1. Any amount above 220 is a fraud.
    2. Each day, 2 terminals are compromised for 28 days: all their transactions are frauds.
    3. Each day, 3 customers are compromised for 14 days: on average a third of their
       transactions are frauds, with an amount multiplied by 5.
    """
    day = tx["tx_time_days"].to_numpy()
    scenario_1 = tx["tx_amount"].to_numpy() > 220
    scenario_2 = compromised(n_terminals, nb_days, 2, 28, rng)[tx["terminal_id"].to_numpy(), day]
    customer_hit = compromised(n_customers, nb_days, 3, 14, rng)[tx["customer_id"].to_numpy(), day]
    scenario_3 = customer_hit & (rng.random(len(tx)) < 1 / 3)

    tx = tx.copy()
    tx["tx_amount"] = np.where(scenario_3, tx["tx_amount"] * 5, tx["tx_amount"]).round(2)
    tx["tx_fraud"] = (scenario_1 | scenario_2 | scenario_3).astype(np.int64)
    tx["tx_fraud_scenario"] = np.select([scenario_3, scenario_2, scenario_1], [3, 2, 1], 0)
    return tx


def generate_dataset(
    n_customers: int = 5_000,
    n_terminals: int = 10_000,
    nb_days: int = 183,
    start_date: str = "2018-04-01",
    radius: float = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Labelled transactions with the book's parameters by default, identical for a given seed."""
    rng = np.random.default_rng(seed)
    customers = customer_profiles(n_customers, rng)
    terminals = terminal_profiles(n_terminals, rng)
    tx = genuine_transactions(customers, terminals, nb_days, start_date, radius, rng)
    return add_frauds(tx, n_customers, n_terminals, nb_days, rng)


def main() -> None:
    tx = generate_dataset()
    TRANSACTIONS_PATH.parent.mkdir(exist_ok=True)
    tx.to_parquet(TRANSACTIONS_PATH, index=False)
    rate = tx["tx_fraud"].mean()
    print(f"{len(tx):,} transactions, {rate:.2%} frauds, saved to {TRANSACTIONS_PATH}")


if __name__ == "__main__":
    main()
