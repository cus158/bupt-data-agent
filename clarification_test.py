"""Focused regression checks for the lightweight clarification gate."""

from __future__ import annotations

import argparse
from unittest.mock import patch

from agent import (
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
    "Q1": "查询2025年上半年各门店的成交销额，并按销额降序排列。",
    "Q2": "查询2025年上半年华东战区即时零售动销最好的3个SKU。",
    "Q3": "计算2025年上半年各门店退损率，找出超过5%的门店，并分析主要退款原因。",
    "Q4": "比较2025年Q1和Q2各门店成交销额，找出增长超过10%的门店。",
    "Q5": "找出Q2相比Q1成交销额增长但毛利率下降的门店，并分析是否可能存在低毛利SKU放量拖累。",
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
        patch("agent.load_config", return_value=LLMConfig("hidden", "test", None)),
        patch("agent.create_llm_client", return_value=object()),
        patch("agent.load_business_context", return_value="business"),
        patch("agent.load_schema_context", return_value="schema"),
        patch("agent.generate_sql_plan", return_value=clarification_plan),
        patch("agent.validate_business_rules") as validate_business_rules,
        patch("agent.execute_query") as execute_query,
        patch("agent.summarize_result") as summarize_result,
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
