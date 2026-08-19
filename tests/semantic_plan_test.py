"""Deterministic parser and repair propagation checks for SemanticPlan."""

from __future__ import annotations

from unittest.mock import patch

from bupt_data_agent.agent import (
    SQLPlan,
    SemanticPlan,
    _parse_sql_plan,
    repair_sql_plan,
)


def _ready_payload() -> dict:
    return {
        "status": "ready",
        "clarification_question": None,
        "ambiguity_type": None,
        "semantic_plan": {
            "intent": "ranking",
            "metrics": ["成交销额", "销量"],
            "dimensions": ["SKU", "品类"],
            "time_range": "2025-H1",
            "filters": ["战区=华东", "渠道=O2O", "Top3"],
            "tables": [
                "sales_order",
                "store_info",
                "product_info",
                "invented_table",
            ],
            "visualization_intent": "none",
        },
        "sql": "SELECT product_id FROM sales_order",
        "reasoning_summary": "按业务条件查询SKU。",
        "chart_type": "none",
    }


def main() -> None:
    normal = _parse_sql_plan(_ready_payload())
    assert normal.semantic_plan == SemanticPlan(
        intent="ranking",
        metrics=("成交销额", "销量"),
        dimensions=("SKU", "品类"),
        time_range="2025-H1",
        filters=("战区=华东", "渠道=O2O", "Top3"),
        tables=("sales_order", "store_info", "product_info"),
        visualization_intent="none",
    )
    print("SemanticPlan normal payload: PASS")

    partial_payload = _ready_payload()
    partial_payload["semantic_plan"] = {"intent": "aggregation"}
    partial = _parse_sql_plan(partial_payload)
    assert partial.semantic_plan == SemanticPlan(intent="aggregation")
    print("SemanticPlan missing optional fields: PASS")

    missing_payload = _ready_payload()
    missing_payload.pop("semantic_plan")
    missing = _parse_sql_plan(missing_payload)
    assert missing.semantic_plan is None and missing.sql
    print("SemanticPlan omitted with safe fallback: PASS")

    clarification = _parse_sql_plan(
        {
            "status": "needs_clarification",
            "clarification_question": "请明确按销量还是成交销额衡量？",
            "ambiguity_type": "metric",
            "semantic_plan": None,
            "sql": None,
            "reasoning_summary": "指标存在歧义。",
            "chart_type": "none",
        }
    )
    assert clarification.status == "needs_clarification"
    assert clarification.sql is None and clarification.semantic_plan is None
    print("SemanticPlan clarification payload: PASS")

    repaired_source = SQLPlan(
        sql="SELECT store_id FROM store_info",
        reasoning_summary="修复后的门店查询。",
        chart_type="bar",
        semantic_plan=SemanticPlan(
            intent="ranking",
            metrics=("成交销额",),
            dimensions=("门店",),
            time_range="2025-H1",
            filters=("按成交销额降序",),
            tables=("sales_order", "store_info"),
            visualization_intent="bar",
        ),
    )
    with patch(
        "bupt_data_agent.agent._call_json_plan",
        return_value=repaired_source,
    ):
        repaired = repair_sql_plan(
            object(),
            "test-model",
            "查询门店销售额并画柱状图。",
            SQLPlan("SELECT 1", "原始查询。", "bar"),
            "test error",
            "schema",
            "business",
        )
    assert repaired.semantic_plan == repaired_source.semantic_plan
    assert repaired.chart_type == "bar"
    print("SemanticPlan repair propagation: PASS")
    print("All SemanticPlan tests passed.")


if __name__ == "__main__":
    main()
