"""Deterministic tests for the Business Rule Validator. No LLM calls are made."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agent import (
    LLMConfig,
    QueryResult,
    SQLPlan,
    load_business_context,
    run_agent,
)
from business_validator import BusinessValidationResult, validate_business_rules


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "outputs" / "online_test_report.txt"
BUSINESS_CONTEXT = load_business_context()


def check(
    name: str,
    question: str,
    sql: str,
    *,
    valid: bool,
    violation: str | None = None,
    warning: str | None = None,
) -> BusinessValidationResult:
    result = validate_business_rules(question, sql, BUSINESS_CONTEXT)
    rules = {issue.rule for issue in result.violations}
    warnings = {issue.rule for issue in result.warnings}
    assert result.valid is valid, (
        f"{name}: expected valid={valid}, got {result.as_dict()}"
    )
    if violation:
        assert violation in rules, f"{name}: missing violation {violation}: {result.as_dict()}"
    if warning:
        assert warning in warnings, f"{name}: missing warning {warning}: {result.as_dict()}"
    print(
        f"Test {name}: PASS | valid={result.valid} | "
        f"violations={sorted(rules) or ['none']} | warnings={sorted(warnings) or ['none']}"
    )
    return result


def run_a_to_j() -> None:
    check(
        "A",
        "查询各门店成交销额",
        """
        SELECT store_id, SUM(quantity * sale_price) AS sales_amount
        FROM sales_order GROUP BY store_id;
        """,
        valid=False,
        violation="transaction_sales_amount",
    )
    check(
        "B",
        "查询各门店成交销额",
        """
        SELECT store_id,
               SUM(quantity * sale_price - discount_amount) AS sales_amount
        FROM sales_order GROUP BY store_id;
        """,
        valid=True,
    )
    check(
        "C",
        "查询各门店毛利率",
        """
        SELECT store_id,
               AVG(((quantity * sale_price - discount_amount)
                    - quantity * unit_cost)
                   / (quantity * sale_price - discount_amount)) AS gross_margin
        FROM sales_order JOIN product_info USING (product_id)
        GROUP BY store_id;
        """,
        valid=False,
        violation="gross_margin_aggregation",
    )
    check(
        "D",
        "查询各门店毛利率",
        "SELECT store_id, SUM(gross_profit) / SUM(sales_amount) AS gross_margin FROM metrics GROUP BY store_id;",
        valid=True,
        warning="division_by_zero",
    )
    check(
        "E",
        "查询2025年上半年退款金额",
        """
        SELECT SUM(r.refund_amount)
        FROM refund_record AS r
        JOIN sales_order AS s ON s.order_id = r.order_id
        WHERE s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01';
        """,
        valid=False,
        violation="refund_time_field",
    )
    check(
        "F",
        "查询2025年上半年退款金额",
        """
        SELECT SUM(refund_amount)
        FROM refund_record
        WHERE refund_date >= '2025-01-01' AND refund_date < '2025-07-01';
        """,
        valid=True,
    )
    check(
        "G",
        "计算各门店退损率",
        """
        SELECT store_id,
               COUNT(refund_id) * 1.0 / COUNT(order_id) AS refund_rate
        FROM refund_record GROUP BY store_id;
        """,
        valid=False,
        violation="refund_loss_rate_formula",
    )
    check(
        "H",
        "计算各门店Q1到Q2成交销额增长率",
        "SELECT store_id, (q2 - q1) / q1 AS growth_rate FROM quarterly_sales;",
        valid=True,
        warning="division_by_zero",
    )
    check(
        "I",
        "查询所有门店",
        "SELECT * FROM dim_store;",
        valid=False,
        violation="real_table_names",
    )
    check(
        "J",
        "查询销量最高的商品",
        """
        SELECT product_id, SUM(quantity) AS sales_quantity
        FROM sales_order GROUP BY product_id ORDER BY sales_quantity DESC;
        """,
        valid=True,
    )

    check(
        "K (join amplification warning)",
        "计算各门店退损率",
        """
        SELECT s.store_id,
               SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount,
               SUM(r.refund_amount) AS refund_amount
        FROM sales_order AS s
        JOIN refund_record AS r ON r.order_id = s.order_id
        GROUP BY s.store_id;
        """,
        valid=True,
        warning="refund_sales_join_amplification",
    )


def _saved_queries() -> list[tuple[str, str, str]]:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Saved online report not found: {REPORT_PATH}")
    report = REPORT_PATH.read_text(encoding="utf-8-sig")
    matches = re.findall(
        r"^Q([1-5])\r?\n用户原问题:\r?\n(.*?)\r?\n第一次 LLM SQL:\r?\n"
        r"(.*?)\r?\nreasoning_summary:",
        report,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(matches) != 5:
        raise AssertionError(f"Expected 5 saved queries, found {len(matches)}")
    return [(f"Q{number}", question.strip(), sql.strip()) for number, question, sql in matches]


def run_saved_q1_to_q5() -> None:
    for name, question, sql in _saved_queries():
        result = validate_business_rules(question, sql, BUSINESS_CONTEXT)
        assert result.valid, f"{name} false positive: {result.as_dict()}"
        warning_rules = [issue.rule for issue in result.warnings]
        assert "refund_sales_join_amplification" not in warning_rules, (
            f"{name} incorrectly flagged detail-join amplification: {result.as_dict()}"
        )
        print(f"Saved {name}: VALID | warnings={warning_rules or ['none']}")


def run_agent_repair_check() -> None:
    bad_sql = (
        "SELECT store_id, SUM(quantity * sale_price) AS sales_amount "
        "FROM sales_order GROUP BY store_id"
    )
    repaired_sql = (
        "SELECT store_id, "
        "SUM(quantity * sale_price - discount_amount) AS sales_amount "
        "FROM sales_order GROUP BY store_id"
    )
    first_plan = SQLPlan(
        sql=bad_sql,
        reasoning_summary="首次 SQL 漏扣优惠金额。",
        chart_type="bar",
    )
    repaired_plan = SQLPlan(
        sql=repaired_sql,
        reasoning_summary="按成交销额公式修复。",
        chart_type="bar",
    )
    query_result = QueryResult(
        dataframe=pd.DataFrame([{"store_id": "S001", "sales_amount": 1.0}]),
        truncated=False,
    )

    with (
        patch("agent.load_config", return_value=LLMConfig("hidden", "test", None)),
        patch("agent.create_llm_client", return_value=object()),
        patch("agent.load_business_context", return_value=BUSINESS_CONTEXT),
        patch("agent.load_schema_context", return_value="schema"),
        patch("agent.generate_sql_plan", return_value=first_plan),
        patch("agent.repair_sql_plan", return_value=repaired_plan) as repair,
        patch("agent.execute_query", return_value=query_result) as execute,
        patch("agent.summarize_result", return_value="done"),
    ):
        result = run_agent("查询各门店成交销额")

    assert result.repair_triggered
    assert result.first_error_type == "BusinessRuleValidationError"
    assert result.business_validation and result.business_validation.valid
    repair.assert_called_once()
    execute.assert_called_once_with(repaired_sql)
    print(
        "Agent repair integration: PASS | invalid first SQL was not executed | "
        "shared repair count=1"
    )


def main() -> int:
    run_a_to_j()
    run_saved_q1_to_q5()
    run_agent_repair_check()
    print("All deterministic Business Rule Validator tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
