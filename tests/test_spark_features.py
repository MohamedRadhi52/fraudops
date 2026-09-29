import pandas as pd
import pytest

from fraudops import features
from fraudops.spark_features import build_features, spark_session


@pytest.fixture(scope="module")
def spark():
    session = spark_session()
    yield session
    session.stop()


def test_spark_and_pandas_give_the_same_features(spark, small_transactions, tmp_path):
    path = tmp_path / "transactions.parquet"
    small_transactions.to_parquet(path, index=False)
    result = build_features(spark.read.parquet(str(path))).toPandas()
    result = result.sort_values("transaction_id", ignore_index=True)

    expected = features.build_features(small_transactions)
    columns = expected.columns.difference(small_transactions.columns)
    pd.testing.assert_frame_equal(result[columns], expected[columns])
