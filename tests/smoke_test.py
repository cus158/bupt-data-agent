"""Deterministic SQLite Golden SQL tests for the five example questions."""

from __future__ import annotations

import math
import json
import sqlite3

from bupt_data_agent.paths import DB_PATH, EVALUATION_DIR

GOLDEN_RESULTS_PATH = EVALUATION_DIR / "golden_results.json"

TEST_1_SQL = """
SELECT
    s.store_id,
    st.store_name,
    ROUND(SUM(s.quantity * s.sale_price - s.discount_amount), 2) AS sales_amount
FROM sales_order AS s
JOIN store_info AS st ON st.store_id = s.store_id
WHERE s.order_date >= '2025-01-01'
  AND s.order_date < '2025-07-01'
GROUP BY s.store_id, st.store_name
ORDER BY sales_amount DESC, s.store_id;
"""

TEST_2_SQL = """
SELECT
    s.product_id,
    p.product_name,
    p.category,
    SUM(s.quantity) AS sales_quantity,
    ROUND(SUM(s.quantity * s.sale_price - s.discount_amount), 2) AS sales_amount
FROM sales_order AS s
JOIN store_info AS st ON st.store_id = s.store_id
JOIN product_info AS p ON p.product_id = s.product_id
WHERE s.order_date >= '2025-01-01'
  AND s.order_date < '2025-07-01'
  AND st.region = '华东'
  AND s.channel_code = 'O2O'
GROUP BY s.product_id, p.product_name, p.category
ORDER BY sales_amount DESC, s.product_id
LIMIT 3;
"""

TEST_3_SQL = """
WITH sales_agg AS (
    SELECT
        store_id,
        SUM(quantity * sale_price - discount_amount) AS sales_amount_sum
    FROM sales_order
    WHERE order_date >= '2025-01-01'
      AND order_date < '2025-07-01'
    GROUP BY store_id
),
refund_agg AS (
    SELECT
        s.store_id,
        SUM(r.refund_amount) AS refund_amount_sum
    FROM refund_record AS r
    JOIN sales_order AS s ON s.order_id = r.order_id
    WHERE r.refund_date >= '2025-01-01'
      AND r.refund_date < '2025-07-01'
    GROUP BY s.store_id
)
SELECT
    sa.store_id,
    st.store_name,
    ROUND(sa.sales_amount_sum, 2) AS sales_amount_sum,
    ROUND(COALESCE(ra.refund_amount_sum, 0), 2) AS refund_amount_sum,
    ROUND(
        COALESCE(ra.refund_amount_sum, 0) / NULLIF(sa.sales_amount_sum, 0),
        6
    ) AS refund_rate
FROM sales_agg AS sa
JOIN store_info AS st ON st.store_id = sa.store_id
LEFT JOIN refund_agg AS ra ON ra.store_id = sa.store_id
ORDER BY refund_rate DESC, sa.store_id;
"""

TEST_4_SQL = """
WITH quarterly_sales AS (
    SELECT
        store_id,
        SUM(
            CASE
                WHEN order_date >= '2025-01-01' AND order_date < '2025-04-01'
                THEN quantity * sale_price - discount_amount
                ELSE 0
            END
        ) AS q1_sales,
        SUM(
            CASE
                WHEN order_date >= '2025-04-01' AND order_date < '2025-07-01'
                THEN quantity * sale_price - discount_amount
                ELSE 0
            END
        ) AS q2_sales
    FROM sales_order
    WHERE order_date >= '2025-01-01'
      AND order_date < '2025-07-01'
    GROUP BY store_id
)
SELECT
    qs.store_id,
    st.store_name,
    ROUND(qs.q1_sales, 2) AS q1_sales,
    ROUND(qs.q2_sales, 2) AS q2_sales,
    CASE
        WHEN qs.q1_sales = 0 THEN NULL
        ELSE ROUND((qs.q2_sales - qs.q1_sales) / qs.q1_sales, 6)
    END AS growth_rate
FROM quarterly_sales AS qs
JOIN store_info AS st ON st.store_id = qs.store_id
ORDER BY growth_rate DESC, qs.store_id;
"""

TEST_3_REFUND_REASONS_SQL = """
WITH sales_agg AS (
    SELECT
        store_id,
        SUM(quantity * sale_price - discount_amount) AS sales_amount_sum
    FROM sales_order
    WHERE order_date >= '2025-01-01'
      AND order_date < '2025-07-01'
    GROUP BY store_id
),
refund_base AS (
    SELECT
        s.store_id,
        r.refund_reason,
        r.refund_amount
    FROM refund_record AS r
    JOIN sales_order AS s ON s.order_id = r.order_id
    WHERE r.refund_date >= '2025-01-01'
      AND r.refund_date < '2025-07-01'
),
refund_store_agg AS (
    SELECT
        store_id,
        SUM(refund_amount) AS refund_amount_sum
    FROM refund_base
    GROUP BY store_id
),
high_refund_stores AS (
    SELECT
        sa.store_id,
        rsa.refund_amount_sum
    FROM sales_agg AS sa
    JOIN refund_store_agg AS rsa ON rsa.store_id = sa.store_id
    WHERE rsa.refund_amount_sum / NULLIF(sa.sales_amount_sum, 0) > 0.05
)
SELECT
    rb.store_id,
    st.store_name,
    rb.refund_reason,
    COUNT(*) AS refund_count,
    ROUND(SUM(rb.refund_amount), 2) AS refund_amount,
    ROUND(
        SUM(rb.refund_amount) / NULLIF(hrs.refund_amount_sum, 0),
        6
    ) AS refund_amount_share
FROM refund_base AS rb
JOIN high_refund_stores AS hrs ON hrs.store_id = rb.store_id
JOIN store_info AS st ON st.store_id = rb.store_id
GROUP BY rb.store_id, st.store_name, rb.refund_reason, hrs.refund_amount_sum
ORDER BY rb.store_id, refund_amount DESC, rb.refund_reason;
"""

TEST_5_STORES_SQL = """
WITH order_metrics AS (
    SELECT
        s.store_id,
        s.order_date,
        s.quantity * s.sale_price - s.discount_amount AS sales_amount,
        s.quantity * s.sale_price - s.discount_amount
            - s.quantity * p.unit_cost AS gross_profit
    FROM sales_order AS s
    JOIN product_info AS p ON p.product_id = s.product_id
    WHERE s.order_date >= '2025-01-01'
      AND s.order_date < '2025-07-01'
),
store_quarterly AS (
    SELECT
        store_id,
        SUM(CASE WHEN order_date < '2025-04-01' THEN sales_amount ELSE 0 END)
            AS q1_sales_amount,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN sales_amount ELSE 0 END)
            AS q2_sales_amount,
        SUM(CASE WHEN order_date < '2025-04-01' THEN gross_profit ELSE 0 END)
            AS q1_gross_profit,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN gross_profit ELSE 0 END)
            AS q2_gross_profit
    FROM order_metrics
    GROUP BY store_id
),
store_metrics AS (
    SELECT
        store_id,
        q1_sales_amount,
        q2_sales_amount,
        (q2_sales_amount - q1_sales_amount) / NULLIF(q1_sales_amount, 0)
            AS sales_growth,
        q1_gross_profit,
        q2_gross_profit,
        q1_gross_profit / NULLIF(q1_sales_amount, 0) AS q1_gross_margin_rate,
        q2_gross_profit / NULLIF(q2_sales_amount, 0) AS q2_gross_margin_rate
    FROM store_quarterly
)
SELECT
    sm.store_id,
    st.store_name,
    ROUND(sm.q1_sales_amount, 2) AS q1_sales_amount,
    ROUND(sm.q2_sales_amount, 2) AS q2_sales_amount,
    ROUND(sm.sales_growth, 6) AS sales_growth,
    ROUND(sm.q1_gross_profit, 2) AS q1_gross_profit,
    ROUND(sm.q2_gross_profit, 2) AS q2_gross_profit,
    ROUND(sm.q1_gross_margin_rate, 6) AS q1_gross_margin_rate,
    ROUND(sm.q2_gross_margin_rate, 6) AS q2_gross_margin_rate,
    ROUND(sm.q2_gross_margin_rate - sm.q1_gross_margin_rate, 6)
        AS gross_margin_rate_change
FROM store_metrics AS sm
JOIN store_info AS st ON st.store_id = sm.store_id
WHERE sm.q2_sales_amount > sm.q1_sales_amount
  AND sm.q2_gross_margin_rate < sm.q1_gross_margin_rate
ORDER BY sm.store_id;
"""

TEST_5_SKUS_SQL = """
WITH order_metrics AS (
    SELECT
        s.store_id,
        s.product_id,
        s.order_date,
        s.quantity,
        s.quantity * s.sale_price - s.discount_amount AS sales_amount,
        s.quantity * s.sale_price - s.discount_amount
            - s.quantity * p.unit_cost AS gross_profit
    FROM sales_order AS s
    JOIN product_info AS p ON p.product_id = s.product_id
    WHERE s.order_date >= '2025-01-01'
      AND s.order_date < '2025-07-01'
),
store_quarterly AS (
    SELECT
        store_id,
        SUM(CASE WHEN order_date < '2025-04-01' THEN sales_amount ELSE 0 END)
            AS q1_sales_amount,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN sales_amount ELSE 0 END)
            AS q2_sales_amount,
        SUM(CASE WHEN order_date < '2025-04-01' THEN gross_profit ELSE 0 END)
            AS q1_gross_profit,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN gross_profit ELSE 0 END)
            AS q2_gross_profit
    FROM order_metrics
    GROUP BY store_id
),
target_stores AS (
    SELECT
        store_id,
        q2_sales_amount
    FROM store_quarterly
    WHERE q2_sales_amount > q1_sales_amount
      AND q2_gross_profit / NULLIF(q2_sales_amount, 0)
          < q1_gross_profit / NULLIF(q1_sales_amount, 0)
),
sku_quarterly AS (
    SELECT
        store_id,
        product_id,
        SUM(CASE WHEN order_date < '2025-04-01' THEN quantity ELSE 0 END)
            AS q1_quantity,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN quantity ELSE 0 END)
            AS q2_quantity,
        SUM(CASE WHEN order_date < '2025-04-01' THEN sales_amount ELSE 0 END)
            AS q1_sales_amount,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN sales_amount ELSE 0 END)
            AS q2_sales_amount,
        SUM(CASE WHEN order_date < '2025-04-01' THEN gross_profit ELSE 0 END)
            AS q1_gross_profit,
        SUM(CASE WHEN order_date >= '2025-04-01' THEN gross_profit ELSE 0 END)
            AS q2_gross_profit
    FROM order_metrics
    GROUP BY store_id, product_id
)
SELECT
    sq.store_id,
    st.store_name,
    sq.product_id,
    p.product_name,
    p.category,
    sq.q1_quantity,
    sq.q2_quantity,
    ROUND(
        (sq.q2_quantity - sq.q1_quantity) * 1.0 / NULLIF(sq.q1_quantity, 0),
        6
    ) AS quantity_growth,
    ROUND(sq.q1_sales_amount, 2) AS q1_sales_amount,
    ROUND(sq.q2_sales_amount, 2) AS q2_sales_amount,
    ROUND(
        (sq.q2_sales_amount - sq.q1_sales_amount)
            / NULLIF(sq.q1_sales_amount, 0),
        6
    ) AS sales_growth,
    ROUND(sq.q1_gross_profit, 2) AS q1_gross_profit,
    ROUND(sq.q2_gross_profit, 2) AS q2_gross_profit,
    ROUND(
        sq.q1_gross_profit / NULLIF(sq.q1_sales_amount, 0),
        6
    ) AS q1_gross_margin_rate,
    ROUND(
        sq.q2_gross_profit / NULLIF(sq.q2_sales_amount, 0),
        6
    ) AS q2_gross_margin_rate,
    ROUND(
        sq.q2_sales_amount / NULLIF(ts.q2_sales_amount, 0),
        6
    ) AS q2_sales_share
FROM sku_quarterly AS sq
JOIN target_stores AS ts ON ts.store_id = sq.store_id
JOIN store_info AS st ON st.store_id = sq.store_id
JOIN product_info AS p ON p.product_id = sq.product_id
ORDER BY sq.store_id, q2_sales_share DESC, sq.product_id;
"""


def connect_read_only() -> sqlite3.Connection:
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. "
            "Run 'python -m bupt_data_agent.prepare_db' first."
        )
    conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def print_result(title: str, sql: str, rows: list[sqlite3.Row]) -> None:
    print(f"\n=== {title} ===")
    print(sql.strip())
    if not rows:
        print("(no rows)")
        return

    headers = rows[0].keys()
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join(str(row[header]) for header in headers))


def assert_close(actual: float, expected: float, tolerance: float = 0.01) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, got {actual}")


def rows_as_dicts(rows: list[sqlite3.Row]) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def test_1(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_1_SQL).fetchall()
    print_result("Test 1 - H1 sales amount by store", TEST_1_SQL, rows)
    assert [row["store_id"] for row in rows] == ["S001", "S003", "S002", "S004"]
    expected = {
        "S001": 1_201_810.63,
        "S002": 915_688.07,
        "S003": 1_056_452.94,
        "S004": 854_841.01,
    }
    for row in rows:
        assert_close(row["sales_amount"], expected[row["store_id"]])
    print("PASS")
    return rows_as_dicts(rows)


def test_2(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_2_SQL).fetchall()
    print_result("Test 2 - East China O2O Top 3 SKU", TEST_2_SQL, rows)
    assert [row["product_id"] for row in rows] == ["P008", "P007", "P001"]
    expected = {
        "P008": (134, 101_287.91),
        "P007": (102, 97_260.10),
        "P001": (268, 75_544.43),
    }
    for row in rows:
        quantity, sales_amount = expected[row["product_id"]]
        assert row["sales_quantity"] == quantity
        assert_close(row["sales_amount"], sales_amount)
    print("PASS")
    return rows_as_dicts(rows)


def test_3(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_3_SQL).fetchall()
    print_result("Test 3 - H1 refund rate by store", TEST_3_SQL, rows)
    assert len(rows) == 4
    assert [row["store_id"] for row in rows if row["refund_rate"] > 0.05] == [
        "S003",
        "S001",
    ]
    expected_refunds = {
        "S001": 68_222.42,
        "S002": 15_731.57,
        "S003": 85_538.23,
        "S004": 22_259.46,
    }
    for row in rows:
        assert_close(row["refund_amount_sum"], expected_refunds[row["store_id"]])
        if row["sales_amount_sum"] == 0 and row["refund_rate"] is not None:
            raise AssertionError("Refund rate must be NULL when sales amount is zero")
    print("PASS")
    return rows_as_dicts(rows)


def test_4(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_4_SQL).fetchall()
    print_result("Test 4 - Q1 to Q2 sales growth", TEST_4_SQL, rows)
    assert len(rows) == 4
    assert [row["store_id"] for row in rows if row["growth_rate"] > 0.10] == [
        "S001",
        "S003",
    ]
    for row in rows:
        if row["q1_sales"] == 0 and row["growth_rate"] is not None:
            raise AssertionError("Growth rate must be NULL when Q1 sales are zero")
    print("PASS")
    return rows_as_dicts(rows)


def test_3_refund_reasons(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_3_REFUND_REASONS_SQL).fetchall()
    print_result(
        "Test 3 detail - refund reasons for stores over 5%",
        TEST_3_REFUND_REASONS_SQL,
        rows,
    )
    assert {row["store_id"] for row in rows} == {"S001", "S003"}
    assert len(rows) == 8
    for store_id in ("S001", "S003"):
        store_rows = [row for row in rows if row["store_id"] == store_id]
        assert_close(
            sum(row["refund_amount_share"] for row in store_rows),
            1.0,
            tolerance=0.00001,
        )
    print("PASS")
    return rows_as_dicts(rows)


def test_5_stores(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_5_STORES_SQL).fetchall()
    print_result(
        "Test 5 - stores with sales growth and gross margin decline",
        TEST_5_STORES_SQL,
        rows,
    )
    assert [row["store_id"] for row in rows] == ["S001", "S002", "S003"]
    for row in rows:
        assert row["q2_sales_amount"] > row["q1_sales_amount"]
        assert row["q2_gross_margin_rate"] < row["q1_gross_margin_rate"]
        assert_close(
            row["gross_margin_rate_change"],
            row["q2_gross_margin_rate"] - row["q1_gross_margin_rate"],
            tolerance=0.000002,
        )
    print("PASS")
    return rows_as_dicts(rows)


def test_5_skus(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(TEST_5_SKUS_SQL).fetchall()
    print_result("Test 5 detail - SKU evidence", TEST_5_SKUS_SQL, rows)
    assert {row["store_id"] for row in rows} == {"S001", "S002", "S003"}
    assert len(rows) == 36
    for store_id in ("S001", "S002", "S003"):
        store_rows = [row for row in rows if row["store_id"] == store_id]
        assert_close(
            sum(row["q2_sales_share"] for row in store_rows),
            1.0,
            tolerance=0.00001,
        )
    print("PASS")
    return rows_as_dicts(rows)


def save_golden_results(results: dict[str, dict[str, object]]) -> None:
    payload = {
        "database": "data/business.db",
        "note": (
            "Deterministic benchmark results. Example 5 provides evidence only and "
            "does not prove that SKU changes caused the store-level margin decline."
        ),
        "queries": results,
    }
    GOLDEN_RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGolden results saved: {GOLDEN_RESULTS_PATH}")


def main() -> None:
    print(f"Database: {DB_PATH}")
    with connect_read_only() as conn:
        results = {
            "example_1_store_sales": {"sql": TEST_1_SQL.strip(), "rows": test_1(conn)},
            "example_2_top_skus": {"sql": TEST_2_SQL.strip(), "rows": test_2(conn)},
            "example_3_refund_rates": {"sql": TEST_3_SQL.strip(), "rows": test_3(conn)},
            "example_3_refund_reasons": {
                "sql": TEST_3_REFUND_REASONS_SQL.strip(),
                "rows": test_3_refund_reasons(conn),
            },
            "example_4_quarterly_growth": {
                "sql": TEST_4_SQL.strip(),
                "rows": test_4(conn),
            },
            "example_5_target_stores": {
                "sql": TEST_5_STORES_SQL.strip(),
                "rows": test_5_stores(conn),
            },
            "example_5_sku_evidence": {
                "sql": TEST_5_SKUS_SQL.strip(),
                "rows": test_5_skus(conn),
            },
        }
    save_golden_results(results)
    print("\nAll Golden SQL smoke tests passed.")


if __name__ == "__main__":
    main()
