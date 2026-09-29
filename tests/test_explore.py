from fraudops.explore import SQL_DIR, connect


def test_every_query_returns_rows(small_transactions, tmp_path):
    path = tmp_path / "transactions.parquet"
    small_transactions.to_parquet(path, index=False)
    con = connect(path)
    queries = sorted(SQL_DIR.glob("*.sql"))
    assert len(queries) >= 5
    for query in queries:
        assert con.sql(query.read_text()).fetchall(), query.name
