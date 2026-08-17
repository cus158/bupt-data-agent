"""Build and validate the SQLite database from the four source workbooks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "business.db"
TEMP_DB_PATH = DATA_DIR / "business.db.tmp"

TABLE_ORDER = ("store_info", "product_info", "sales_order", "refund_record")

TABLE_CONFIG = {
    "store_info": {
        "file": "store_info.xlsx",
        "columns": ["store_id", "store_name", "region", "city", "open_date"],
        "text": ["store_id", "store_name", "region", "city"],
        "dates": ["open_date"],
        "integers": [],
        "reals": [],
        "primary_key": "store_id",
    },
    "product_info": {
        "file": "product_info.xlsx",
        "columns": ["product_id", "product_name", "category", "unit_cost", "list_price"],
        "text": ["product_id", "product_name", "category"],
        "dates": [],
        "integers": [],
        "reals": ["unit_cost", "list_price"],
        "primary_key": "product_id",
    },
    "sales_order": {
        "file": "sales_order.xlsx",
        "columns": [
            "order_id",
            "order_date",
            "store_id",
            "product_id",
            "quantity",
            "sale_price",
            "discount_amount",
            "channel_code",
        ],
        "text": ["order_id", "store_id", "product_id", "channel_code"],
        "dates": ["order_date"],
        "integers": ["quantity"],
        "reals": ["sale_price", "discount_amount"],
        "primary_key": "order_id",
    },
    "refund_record": {
        "file": "refund_record.xlsx",
        "columns": [
            "refund_id",
            "order_id",
            "refund_date",
            "refund_quantity",
            "refund_amount",
            "refund_reason",
        ],
        "text": ["refund_id", "order_id", "refund_reason"],
        "dates": ["refund_date"],
        "integers": ["refund_quantity"],
        "reals": ["refund_amount"],
        "primary_key": "refund_id",
    },
}

EXPECTED_ROW_COUNTS = {
    "store_info": 4,
    "product_info": 12,
    "sales_order": 5980,
    "refund_record": 363,
}

SCHEMA_SQL = """
CREATE TABLE store_info (
    store_id TEXT NOT NULL PRIMARY KEY,
    store_name TEXT NOT NULL,
    region TEXT NOT NULL,
    city TEXT NOT NULL,
    open_date TEXT NOT NULL
);

CREATE TABLE product_info (
    product_id TEXT NOT NULL PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_cost REAL NOT NULL,
    list_price REAL NOT NULL
);

CREATE TABLE sales_order (
    order_id TEXT NOT NULL PRIMARY KEY,
    order_date TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    sale_price REAL NOT NULL,
    discount_amount REAL NOT NULL,
    channel_code TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES store_info(store_id),
    FOREIGN KEY (product_id) REFERENCES product_info(product_id)
);

CREATE TABLE refund_record (
    refund_id TEXT NOT NULL PRIMARY KEY,
    order_id TEXT NOT NULL,
    refund_date TEXT NOT NULL,
    refund_quantity INTEGER NOT NULL,
    refund_amount REAL NOT NULL,
    refund_reason TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES sales_order(order_id)
);
"""

INDEX_SQL = """
CREATE INDEX idx_sales_order_order_date ON sales_order(order_date);
CREATE INDEX idx_sales_order_store_id ON sales_order(store_id);
CREATE INDEX idx_sales_order_product_id ON sales_order(product_id);
CREATE INDEX idx_sales_order_channel_code ON sales_order(channel_code);
CREATE INDEX idx_refund_record_refund_date ON refund_record(refund_date);
CREATE INDEX idx_refund_record_order_id ON refund_record(order_id);
"""

FOREIGN_KEY_CHECKS = {
    "sales_order.store_id -> store_info.store_id": """
        SELECT COUNT(*)
        FROM sales_order AS s
        LEFT JOIN store_info AS st ON st.store_id = s.store_id
        WHERE st.store_id IS NULL
    """,
    "sales_order.product_id -> product_info.product_id": """
        SELECT COUNT(*)
        FROM sales_order AS s
        LEFT JOIN product_info AS p ON p.product_id = s.product_id
        WHERE p.product_id IS NULL
    """,
    "refund_record.order_id -> sales_order.order_id": """
        SELECT COUNT(*)
        FROM refund_record AS r
        LEFT JOIN sales_order AS s ON s.order_id = r.order_id
        WHERE s.order_id IS NULL
    """,
}


def load_workbooks() -> dict[str, pd.DataFrame]:
    """Read and normalize the source workbooks without modifying them."""
    frames: dict[str, pd.DataFrame] = {}

    for table_name in TABLE_ORDER:
        config = TABLE_CONFIG[table_name]
        path = DATA_DIR / config["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source workbook: {path}")

        workbook = pd.ExcelFile(path, engine="openpyxl")
        if workbook.sheet_names != [table_name]:
            raise ValueError(
                f"{path.name} must contain exactly one sheet named {table_name!r}; "
                f"found {workbook.sheet_names!r}"
            )

        frame = pd.read_excel(path, sheet_name=table_name, engine="openpyxl")
        expected_columns = config["columns"]
        if frame.columns.tolist() != expected_columns:
            raise ValueError(
                f"Column mismatch in {path.name}. "
                f"Expected {expected_columns!r}, found {frame.columns.tolist()!r}"
            )
        if frame.isna().any().any():
            null_counts = frame.isna().sum()
            raise ValueError(
                f"Null values found in {path.name}: "
                f"{null_counts[null_counts > 0].to_dict()}"
            )

        for column in config["text"]:
            frame[column] = frame[column].astype("string")
            if (frame[column] != frame[column].str.strip()).any():
                raise ValueError(f"Leading/trailing whitespace found in {table_name}.{column}")
            frame[column] = frame[column].astype(str)

        for column in config["dates"]:
            values = pd.to_datetime(frame[column], errors="raise")
            frame[column] = values.dt.strftime("%Y-%m-%d")

        for column in config["integers"]:
            values = pd.to_numeric(frame[column], errors="raise")
            if (values % 1 != 0).any():
                raise ValueError(f"Non-integer value found in {table_name}.{column}")
            frame[column] = values.astype("int64")

        for column in config["reals"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

        primary_key = config["primary_key"]
        if frame[primary_key].duplicated().any():
            duplicates = frame.loc[frame[primary_key].duplicated(keep=False), primary_key]
            raise ValueError(
                f"Duplicate primary key values in {table_name}.{primary_key}: "
                f"{duplicates.tolist()}"
            )

        expected_count = EXPECTED_ROW_COUNTS[table_name]
        if len(frame) != expected_count:
            raise ValueError(
                f"Unexpected row count for {table_name}: {len(frame)}; "
                f"expected {expected_count}"
            )

        frames[table_name] = frame

    return frames


def insert_frame(conn: sqlite3.Connection, table_name: str, frame: pd.DataFrame) -> None:
    columns = frame.columns.tolist()
    column_sql = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})"
    conn.executemany(sql, frame.itertuples(index=False, name=None))


def validate_database(conn: sqlite3.Connection) -> None:
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")

    foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise RuntimeError(f"SQLite foreign key check failed: {foreign_key_errors}")

    for table_name, expected_count in EXPECTED_ROW_COUNTS.items():
        actual_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if actual_count != expected_count:
            raise RuntimeError(
                f"Row count mismatch for {table_name}: {actual_count}; expected {expected_count}"
            )

    for description, sql in FOREIGN_KEY_CHECKS.items():
        mismatch_count = conn.execute(sql).fetchone()[0]
        if mismatch_count != 0:
            raise RuntimeError(f"Foreign key mismatch for {description}: {mismatch_count}")


def build_database(frames: dict[str, pd.DataFrame]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_DB_PATH.exists():
        TEMP_DB_PATH.unlink()

    try:
        conn = sqlite3.connect(TEMP_DB_PATH)
        try:
            with conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript(SCHEMA_SQL)
                for table_name in TABLE_ORDER:
                    insert_frame(conn, table_name, frames[table_name])
                conn.executescript(INDEX_SQL)
                validate_database(conn)
        finally:
            conn.close()

        TEMP_DB_PATH.replace(DB_PATH)
    except Exception:
        if TEMP_DB_PATH.exists():
            TEMP_DB_PATH.unlink()
        raise


def print_query_rows(cursor: sqlite3.Cursor) -> None:
    headers = [column[0] for column in cursor.description]
    print(" | ".join(headers))
    for row in cursor.fetchall():
        print(" | ".join(str(value) for value in row))


def print_validation_report() -> None:
    print(f"\nDatabase created: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        print("\n=== Row counts ===")
        for table_name in TABLE_ORDER:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"{table_name} = {count}")

        print("\n=== PRAGMA table_info ===")
        for table_name in TABLE_ORDER:
            print(f"\n[{table_name}]")
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            print_query_rows(cursor)

        print("\n=== First 3 rows ===")
        for table_name in TABLE_ORDER:
            print(f"\n[{table_name}]")
            cursor = conn.execute(f"SELECT * FROM {table_name} ORDER BY rowid LIMIT 3")
            print_query_rows(cursor)

        print("\n=== Primary key duplicate checks ===")
        for table_name in TABLE_ORDER:
            primary_key = TABLE_CONFIG[table_name]["primary_key"]
            duplicate_count = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT {primary_key}
                    FROM {table_name}
                    GROUP BY {primary_key}
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            print(f"{table_name}.{primary_key}: {duplicate_count}")

        print("\n=== Foreign key mismatch checks ===")
        for description, sql in FOREIGN_KEY_CHECKS.items():
            mismatch_count = conn.execute(sql).fetchone()[0]
            print(f"{description}: {mismatch_count}")
        pragma_mismatches = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        print(f"PRAGMA foreign_key_check: {pragma_mismatches}")

        print("\n=== Date ranges ===")
        sales_range = conn.execute(
            "SELECT MIN(order_date), MAX(order_date) FROM sales_order"
        ).fetchone()
        refund_range = conn.execute(
            "SELECT MIN(refund_date), MAX(refund_date) FROM refund_record"
        ).fetchone()
        print(f"sales_order.order_date: {sales_range[0]} -> {sales_range[1]}")
        print(f"refund_record.refund_date: {refund_range[0]} -> {refund_range[1]}")

        print("\n=== SQLite checks ===")
        print(f"PRAGMA integrity_check: {conn.execute('PRAGMA integrity_check').fetchone()[0]}")
        print("All database validations passed.")


def main() -> None:
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Source data directory: {DATA_DIR}")
    frames = load_workbooks()
    build_database(frames)
    print_validation_report()


if __name__ == "__main__":
    main()
