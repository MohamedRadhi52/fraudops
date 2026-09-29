"""Run the DuckDB queries of sql/ on the simulated transactions and print their results."""

from pathlib import Path

import duckdb

from fraudops.simulate import TRANSACTIONS_PATH

SQL_DIR = Path("sql")


def connect(transactions_path: Path = TRANSACTIONS_PATH) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection where the Parquet file is the `transactions` view."""
    con = duckdb.connect()
    con.read_parquet(str(transactions_path)).create_view("transactions")
    return con


def main() -> None:
    con = connect()
    for query in sorted(SQL_DIR.glob("*.sql")):
        print(query.name)
        con.sql(query.read_text()).show()


if __name__ == "__main__":
    main()
