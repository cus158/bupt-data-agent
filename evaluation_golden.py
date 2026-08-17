"""Human-auditable Golden SQL for the extended evaluation query cases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "data" / "business.db"

GOLDEN_QUERIES: dict[str, str] = {
    "A1": """
        SELECT COUNT(*) AS order_count
        FROM sales_order
        WHERE order_date >= '2025-05-01' AND order_date < '2025-06-01'
    """,
    "A2": """
        SELECT channel_code,
               SUM(quantity * sale_price - discount_amount) AS sales_amount
        FROM sales_order
        WHERE order_date >= '2025-01-01' AND order_date < '2025-07-01'
        GROUP BY channel_code
        ORDER BY channel_code
    """,
    "A3": """
        SELECT s.product_id, p.product_name,
               SUM(s.quantity) AS sales_quantity
        FROM sales_order AS s
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01'
        GROUP BY s.product_id, p.product_name
        ORDER BY sales_quantity DESC, s.product_id
        LIMIT 5
    """,
    "B1": """
        SELECT p.category,
               SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount
        FROM sales_order AS s
        JOIN store_info AS st ON st.store_id = s.store_id
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE st.region = '华东'
          AND s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01'
        GROUP BY p.category
        ORDER BY sales_amount DESC, p.category
        LIMIT 1
    """,
    "B2": """
        SELECT s.store_id, st.store_name,
               SUM(s.quantity * s.sale_price - s.discount_amount
                   - s.quantity * p.unit_cost)
               / NULLIF(SUM(s.quantity * s.sale_price - s.discount_amount), 0)
               AS gross_margin_rate
        FROM sales_order AS s
        JOIN product_info AS p ON p.product_id = s.product_id
        JOIN store_info AS st ON st.store_id = s.store_id
        WHERE s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01'
        GROUP BY s.store_id, st.store_name
        ORDER BY s.store_id
    """,
    "B3": """
        SELECT s.store_id, st.store_name,
               SUM(r.refund_amount) AS refund_amount
        FROM refund_record AS r
        JOIN sales_order AS s ON s.order_id = r.order_id
        JOIN store_info AS st ON st.store_id = s.store_id
        WHERE r.refund_reason = '质量问题'
        GROUP BY s.store_id, st.store_name
        ORDER BY refund_amount DESC, s.store_id
        LIMIT 1
    """,
    "C1": """
        SELECT s.product_id, p.product_name,
               SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount
        FROM sales_order AS s
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE s.channel_code = 'O2O'
          AND s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01'
        GROUP BY s.product_id, p.product_name
        ORDER BY sales_amount DESC, s.product_id
        LIMIT 3
    """,
    "C2": """
        SELECT s.product_id, p.product_name,
               SUM(s.quantity) AS sales_quantity
        FROM sales_order AS s
        JOIN store_info AS st ON st.store_id = s.store_id
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE st.region = '华东' AND s.channel_code = 'POS'
        GROUP BY s.product_id, p.product_name
        ORDER BY sales_quantity DESC, s.product_id
        LIMIT 1
    """,
    "C3": """
        SELECT s.store_id, st.store_name,
               SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount
        FROM sales_order AS s
        JOIN store_info AS st ON st.store_id = s.store_id
        WHERE s.channel_code = 'B2B'
          AND s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01'
        GROUP BY s.store_id, st.store_name
        ORDER BY sales_amount DESC, s.store_id
        LIMIT 1
    """,
    "D1": """
        SELECT store_id,
               SUM(CASE WHEN order_date >= '2025-01-01'
                              AND order_date < '2025-04-01'
                        THEN quantity * sale_price - discount_amount ELSE 0 END)
                   AS q1_sales,
               SUM(CASE WHEN order_date >= '2025-04-01'
                              AND order_date < '2025-07-01'
                        THEN quantity * sale_price - discount_amount ELSE 0 END)
                   AS q2_sales
        FROM sales_order
        WHERE order_date >= '2025-01-01' AND order_date < '2025-07-01'
        GROUP BY store_id
        ORDER BY store_id
    """,
    "D2": """
        WITH quarter_sales AS (
            SELECT store_id,
                   SUM(CASE WHEN order_date >= '2025-01-01'
                                  AND order_date < '2025-04-01'
                            THEN quantity * sale_price - discount_amount ELSE 0 END)
                       AS q1_sales,
                   SUM(CASE WHEN order_date >= '2025-04-01'
                                  AND order_date < '2025-07-01'
                            THEN quantity * sale_price - discount_amount ELSE 0 END)
                       AS q2_sales
            FROM sales_order
            WHERE order_date >= '2025-01-01' AND order_date < '2025-07-01'
            GROUP BY store_id
        )
        SELECT store_id, q1_sales, q2_sales
        FROM quarter_sales
        WHERE q2_sales < q1_sales
        ORDER BY store_id
    """,
    "D3": """
        SELECT p.category,
               SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount
        FROM sales_order AS s
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE s.order_date >= '2025-04-01' AND s.order_date < '2025-07-01'
        GROUP BY p.category
        ORDER BY p.category
    """,
    "E1": """
        SELECT s.store_id, st.store_name,
               SUM(r.refund_amount) AS refund_amount
        FROM refund_record AS r
        JOIN sales_order AS s ON s.order_id = r.order_id
        JOIN store_info AS st ON st.store_id = s.store_id
        WHERE r.refund_date >= '2025-01-01' AND r.refund_date < '2025-07-01'
        GROUP BY s.store_id, st.store_name
        ORDER BY refund_amount DESC, s.store_id
        LIMIT 1
    """,
    "E2": """
        WITH sales_agg AS (
            SELECT store_id,
                   SUM(quantity * sale_price - discount_amount) AS sales_amount
            FROM sales_order
            WHERE order_date >= '2025-01-01' AND order_date < '2025-07-01'
            GROUP BY store_id
        ),
        refund_agg AS (
            SELECT s.store_id, SUM(r.refund_amount) AS refund_amount
            FROM refund_record AS r
            JOIN sales_order AS s ON s.order_id = r.order_id
            WHERE r.refund_date >= '2025-01-01' AND r.refund_date < '2025-07-01'
            GROUP BY s.store_id
        )
        SELECT sa.store_id,
               COALESCE(ra.refund_amount, 0) / NULLIF(sa.sales_amount, 0)
                   AS refund_rate
        FROM sales_agg AS sa
        LEFT JOIN refund_agg AS ra ON ra.store_id = sa.store_id
        ORDER BY sa.store_id
    """,
    "E3": """
        SELECT SUM(CASE WHEN refund_reason = '质量问题'
                        THEN refund_amount ELSE 0 END)
               / NULLIF(SUM(refund_amount), 0) AS refund_share
        FROM refund_record
        WHERE refund_date >= '2025-01-01' AND refund_date < '2025-07-01'
    """,
    "F4": """
        SELECT s.product_id, p.product_name,
               SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount
        FROM sales_order AS s
        JOIN store_info AS st ON st.store_id = s.store_id
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE st.region = '华东'
        GROUP BY s.product_id, p.product_name
        ORDER BY sales_amount DESC, s.product_id
        LIMIT 1
    """,
}


def get_golden_dataframe(case_id: str) -> pd.DataFrame:
    if case_id not in GOLDEN_QUERIES:
        raise KeyError(f"No Golden SQL for case {case_id}")
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    connection = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(GOLDEN_QUERIES[case_id], connection)
    finally:
        connection.close()

