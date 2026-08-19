"""Focused regression checks for the lightweight clarification gate."""

from __future__ import annotations

import argparse
from unittest.mock import patch

from bupt_data_agent.agent import (
    LLMConfig,
    LLMResponseError,
    SQLPlan,
    _parse_sql_plan,
    create_llm_client,
    generate_sql_plan,
    load_business_context,
    load_config,
    load_schema_context,
    run_agent,
)


AMBIGUOUS_CASES = {
    "A": "哪个商品表现最好？",
    "B": "哪个门店表现最好？",
    "C": "帮我看看华东表现怎么样。",
}

READY_CASES = {
    "D": "华东战区哪个SKU成交销额最高？",
    "E": "即时零售销量最高的5个SKU。",
    "Q1": "查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    "Q2": "查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。",
    "Q3": "查询各门店 2025 年上半年的退损情况，找出退损率超过 5% 的门店，画出各门店退损率对比图，并分析退损率较高门店的主要退款原因。",
    "Q4": "比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。",
    "Q5": "找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。",
}


def run_offline_checks() -> None:
    ready = _parse_sql_plan(
        {
            "status": "ready",
            "clarification_question": None,
            "ambiguity_type": None,
            "sql": "SELECT store_id FROM store_info",
            "reasoning_summary": "使用门店表查询门店编号。",
            "chart_type": "none",
        }
    )
    assert ready.status == "ready" and ready.sql

    clarification = _parse_sql_plan(
        {
            "status": "needs_clarification",
            "clarification_question": "请问希望按销量还是成交销额衡量？",
            "ambiguity_type": "metric",
            "sql": None,
            "reasoning_summary": "缺少用于排名的业务指标。",
            "chart_type": "none",
        }
    )
    assert clarification.sql is None

    invalid_payload = {
        "status": "needs_clarification",
        "clarification_question": "请补充指标。",
        "ambiguity_type": "metric",
        "sql": "SELECT 1",
        "reasoning_summary": "指标不明确。",
        "chart_type": "none",
    }
    try:
        _parse_sql_plan(invalid_payload)
    except LLMResponseError:
        pass
    else:
        raise AssertionError("Clarification payload with SQL should be rejected")

    clarification_plan = SQLPlan(
        sql=None,
        reasoning_summary="缺少排名指标。",
        chart_type="none",
        status="needs_clarification",
        clarification_question="请问希望按销量还是成交销额衡量？",
        ambiguity_type="metric",
    )
    with (
        patch(
            "bupt_data_agent.agent.load_config",
            return_value=LLMConfig("hidden", "test", None),
        ),
        patch("bupt_data_agent.agent.create_llm_client", return_value=object()),
        patch("bupt_data_agent.agent.load_business_context", return_value="business"),
        patch("bupt_data_agent.agent.load_schema_context", return_value="schema"),
        patch(
            "bupt_data_agent.agent.generate_sql_plan",
            return_value=clarification_plan,
        ),
        patch(
            "bupt_data_agent.agent.validate_business_rules"
        ) as validate_business_rules,
        patch("bupt_data_agent.agent.execute_query") as execute_query,
        patch("bupt_data_agent.agent.summarize_result") as summarize_result,
    ):
        result = run_agent("哪个商品表现最好？")
    assert result.plan.status == "needs_clarification"
    assert result.query_result is None and result.conclusion is None
    execute_query.assert_not_called()
    validate_business_rules.assert_not_called()
    summarize_result.assert_not_called()
    print("offline structure and no-SQL-execution checks: PASS")


def run_online_checks() -> None:
    config = load_config()
    client = create_llm_client(config)
    schema_context = load_schema_context()
    business_context = load_business_context()

    failures: list[str] = []
    for name, question in {**AMBIGUOUS_CASES, **READY_CASES}.items():
        plan = generate_sql_plan(
            client,
            config.model,
            question,
            schema_context,
            business_context,
        )
        expected = (
            "needs_clarification" if name in AMBIGUOUS_CASES else "ready"
        )
        sql_state = "null" if plan.sql is None else "present"
        print(
            f"{name}: status={plan.status}, sql={sql_state}, "
            f"ambiguity_type={plan.ambiguity_type or 'none'}"
        )
        if plan.clarification_question:
            print(f"  clarification: {plan.clarification_question}")
        if plan.status != expected:
            failures.append(f"{name}: expected {expected}, got {plan.status}")
        if expected == "needs_clarification" and plan.sql is not None:
            failures.append(f"{name}: clarification unexpectedly returned SQL")
        if expected == "ready" and not plan.sql:
            failures.append(f"{name}: ready plan did not return SQL")

    if failures:
        raise AssertionError("; ".join(failures))
    print("online ambiguity and ready-state checks: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online",
        action="store_true",
        help="Also call the configured LLM once for each regression question.",
    )
    args = parser.parse_args()
    run_offline_checks()
    if args.online:
        run_online_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
