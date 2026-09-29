"""The rolling-window features of fraudops.features, computed with PySpark window functions."""

from pathlib import Path
from time import perf_counter

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.window import WindowSpec

from fraudops import features
from fraudops.features import DELAY, WINDOWS
from fraudops.simulate import SECONDS_PER_DAY, TRANSACTIONS_PATH

SPARK_FEATURES_PATH = Path("data/features_spark")


def spark_session() -> SparkSession:
    """Local session on every core of the machine."""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("fraudops")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def window(key: str, start_days: int, end_days: int) -> WindowSpec:
    """Transactions of the same key made between t - start_days (excluded) and t - end_days."""
    return (
        Window.partitionBy(key)
        .orderBy("tx_time_seconds")
        .rangeBetween(-start_days * SECONDS_PER_DAY + 1, -end_days * SECONDS_PER_DAY)
    )


def build_features(tx: DataFrame) -> DataFrame:
    """Transactions with the same customer and terminal features as the pandas version."""
    columns = []
    for days in WINDOWS:
        customer = window("customer_id", days, 0)
        columns += [
            F.count("*").over(customer).alias(f"customer_nb_tx_{days}d"),
            F.avg("tx_amount").over(customer).alias(f"customer_avg_amount_{days}d"),
        ]
    for days in WINDOWS:
        terminal = window("terminal_id", DELAY + days, DELAY)
        count = F.count("*").over(terminal)
        risk = F.when(count > 0, F.sum("tx_fraud").over(terminal) / count).otherwise(0.0)
        columns += [
            count.alias(f"terminal_nb_tx_{days}d"),
            risk.alias(f"terminal_risk_{days}d"),
        ]
    return tx.select("*", *columns)


def main() -> None:
    """Time both versions on the full dataset and check that they agree."""
    tx = pd.read_parquet(TRANSACTIONS_PATH)
    start = perf_counter()
    expected = features.build_features(tx)
    pandas_seconds = perf_counter() - start

    start = perf_counter()
    spark = spark_session()
    startup_seconds = perf_counter() - start
    start = perf_counter()
    spark_tx = spark.read.parquet(str(TRANSACTIONS_PATH))
    build_features(spark_tx).write.parquet(str(SPARK_FEATURES_PATH), mode="overwrite")
    spark_seconds = perf_counter() - start
    spark.stop()

    result = pd.read_parquet(SPARK_FEATURES_PATH).sort_values("transaction_id", ignore_index=True)
    columns = expected.columns.difference(tx.columns)
    pd.testing.assert_frame_equal(result[columns], expected[columns])
    print(f"pandas: {pandas_seconds:.1f} s")
    print(f"Spark: {startup_seconds:.1f} s to start, then {spark_seconds:.1f} s")
    print(f"Same features for the {len(tx):,} transactions")


if __name__ == "__main__":
    main()
