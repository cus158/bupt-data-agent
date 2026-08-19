"""Offline and optional online tests for bounded conversational follow-ups."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pandas as pd

from bupt_data_agent.agent import (
    LLMConfig,
    QueryResult,
    SQLPlan,
    load_business_context,
    run_agent,
)
from bupt_data_agent.conversation import (
    MAX_ENTITIES_PER_TYPE,
    extract_turn_context,
    format_conversation_context,
    resolve_conversation_context,
)


def _store_context() -> dict:
    return extract_turn_context(
        "2025年上半年哪个门店成交销额最高？",
        """
        SELECT store_id, store_name,
               SUM(quantity * sale_price - discount_amount) AS sales_amount
        FROM sales_order
        WHERE order_date >= '2025-01-01' AND order_date < '2025-07-01'
        GROUP BY store_id ORDER BY sales_amount DESC LIMIT 1
        """,
        pd.DataFrame(
            [{"store_id": "S001", "store_name": "星河路店", "sales_amount": 1.0}]
        ),
    )


def _product_context(count: int = 3) -> dict:
    products = [
        {"product_id": f"P{index:03d}", "product_name": f"商品{index}"}
        for index in range(1, count + 1)
    ]
    return extract_turn_context(
        "华东即时零售成交销额最高的SKU。",
        "SELECT product_id, product_name, SUM(quantity * sale_price - discount_amount) AS sales_amount FROM sales_order GROUP BY product_id",
        pd.DataFrame(products),
    )


def run_offline_tests() -> None:
    # Local plural references can point to an entity set defined earlier in the same question.
    local_reference = resolve_conversation_context(
        "找出增长超过10%且毛利率下降的门店，并分析这些门店的SKU。",
        None,
    )
    assert not local_reference.use_context
    assert local_reference.reference_mode is None
    assert local_reference.clarification_question is None
    print("Local current-question plural reference: PASS")

    # Without a local antecedent or previous context, the same plural expression must clarify.
    missing_plural_context = resolve_conversation_context(
        "这些门店Q2毛利率怎么样？",
        None,
    )
    assert not missing_plural_context.use_context
    assert missing_plural_context.reference_mode == "plural"
    assert missing_plural_context.clarification_question
    print("Missing plural context clarification: PASS")

    # Case 1: one store can satisfy a singular reference.
    store_context = _store_context()
    case1 = resolve_conversation_context("那它Q2的毛利率是多少？", store_context)
    assert case1.use_context and case1.clarification_question is None
    assert "S001" in (case1.prompt_context or "")
    assert store_context["time_context"] == {"year": 2025, "period": "H1"}
    print("Case 1 single-store reference: PASS")

    # Case 2: one SKU can satisfy a singular reference.
    single_product = _product_context(1)
    case2 = resolve_conversation_context("它的销量是多少？", single_product)
    assert case2.use_context and "P001" in (case2.prompt_context or "")
    print("Case 2 single-SKU reference: PASS")

    # Case 3: multiple products plus a singular pronoun must clarify without SQL.
    multiple_products = _product_context(3)
    case3 = resolve_conversation_context("它的毛利率呢？", multiple_products)
    assert not case3.use_context and case3.prompt_context is None
    assert case3.clarification_question
    assert all(product_id in case3.clarification_question for product_id in ("P001", "P002", "P003"))
    with patch("bupt_data_agent.agent.load_config") as load_config:
        result = run_agent("它的毛利率呢？", conversation_context=multiple_products)
    load_config.assert_not_called()
    assert result.plan.status == "needs_clarification" and result.plan.sql is None
    print("Case 3 multi-entity singular clarification: PASS")

    # Case 4: plural reference can inherit all prior stores.
    two_store_context = extract_turn_context(
        "找出退损率超过5%的门店。",
        "SELECT store_id, store_name, refund_rate FROM metrics",
        pd.DataFrame(
            [
                {"store_id": "S003", "store_name": "科技园店", "refund_rate": 0.08},
                {"store_id": "S001", "store_name": "星河路店", "refund_rate": 0.06},
            ]
        ),
    )
    case4 = resolve_conversation_context("它们Q2成交销额分别是多少？", two_store_context)
    assert case4.use_context
    assert all(store_id in (case4.prompt_context or "") for store_id in ("S003", "S001"))
    print("Case 4 multi-entity plural reference: PASS")

    explicit_plural = resolve_conversation_context(
        "这些门店Q2毛利率怎么样？",
        two_store_context,
    )
    assert explicit_plural.use_context
    assert explicit_plural.reference_mode == "plural"
    assert all(
        store_id in (explicit_plural.prompt_context or "")
        for store_id in ("S003", "S001")
    )
    print("Case 4b explicit plural store reference: PASS")

    # Case 5: a complete new question ignores an unrelated previous context.
    case5 = resolve_conversation_context("查询S003 2025年Q2毛利率。", store_context)
    assert not case5.use_context and case5.prompt_context is None
    independent_plan = SQLPlan(
        sql="""
        SELECT s.store_id,
               SUM(s.quantity * s.sale_price - s.discount_amount
                   - s.quantity * p.unit_cost)
               / NULLIF(SUM(s.quantity * s.sale_price - s.discount_amount), 0)
                   AS gross_margin_rate
        FROM sales_order AS s
        JOIN product_info AS p ON p.product_id = s.product_id
        WHERE s.store_id = 'S003'
          AND s.order_date >= '2025-04-01' AND s.order_date < '2025-07-01'
        GROUP BY s.store_id
        """,
        reasoning_summary="只使用当前问题中的 S003、2025 Q2 和毛利率。",
        chart_type="none",
    )
    query_result = QueryResult(
        pd.DataFrame([{"store_id": "S003", "gross_margin_rate": 0.24}]),
        False,
    )

    def generated_plan(*args, **kwargs):
        assert args[5] is None
        return independent_plan

    with (
        patch(
            "bupt_data_agent.agent.load_config",
            return_value=LLMConfig("hidden", "test", None),
        ),
        patch("bupt_data_agent.agent.create_llm_client", return_value=object()),
        patch(
            "bupt_data_agent.agent.load_business_context",
            return_value=load_business_context(),
        ),
        patch("bupt_data_agent.agent.load_schema_context", return_value="schema"),
        patch("bupt_data_agent.agent.generate_sql_plan", side_effect=generated_plan),
        patch("bupt_data_agent.agent.execute_query", return_value=query_result),
        patch(
            "bupt_data_agent.agent.summarize_result",
            return_value="S003 Q2 毛利率。",
        ),
    ):
        independent_result = run_agent(
            "查询S003 2025年Q2毛利率。",
            conversation_context=store_context,
        )
    assert not independent_result.conversation_context_used
    assert independent_result.turn_context["entities"]["stores"][0]["store_id"] == "S003"
    print("Case 5 explicit new question overrides context: PASS")

    # Case 6: after clearing context, an unresolved pronoun must clarify locally.
    case6 = resolve_conversation_context("它Q2毛利率呢？", None)
    assert case6.clarification_question and not case6.use_context
    with patch("bupt_data_agent.agent.load_config") as load_config:
        cleared_result = run_agent("它Q2毛利率呢？", conversation_context=None)
    load_config.assert_not_called()
    assert cleared_result.plan.status == "needs_clarification"
    print("Case 6 cleared/missing context: PASS")

    # Bounded extraction and prompt size.
    large_context = _product_context(MAX_ENTITIES_PER_TYPE + 2)
    assert len(large_context["entities"]["products"]) == MAX_ENTITIES_PER_TYPE
    assert large_context["entity_counts"]["products"] == MAX_ENTITIES_PER_TYPE + 2
    assert large_context["entities_truncated"]["products"]
    assert len(format_conversation_context(large_context)) < 4000
    print("Context entity/token bounds: PASS")


def _assert_ready(result) -> None:
    assert result.plan.status == "ready" and result.plan.sql
    assert result.query_result is not None and not result.query_result.dataframe.empty
    assert result.business_validation and result.business_validation.valid


def run_online_tests() -> None:
    # Online 1: dynamic single-store follow-up.
    first = run_agent("2025年上半年哪个门店成交销额最高？")
    _assert_ready(first)
    context = first.turn_context
    store_id = context["entities"]["stores"][0]["store_id"]
    assert context["entity_counts"]["stores"] == 1 and store_id == "S001"
    follow_up = run_agent("那它Q2的毛利率怎么样？", conversation_context=context)
    _assert_ready(follow_up)
    assert follow_up.conversation_context_used
    assert store_id in follow_up.plan.sql
    assert all(date in follow_up.plan.sql for date in ("2025-04-01", "2025-07-01"))
    assert set(follow_up.query_result.dataframe["store_id"].astype(str)) == {store_id}
    print(f"Online 1: PASS | inherited store={store_id}")

    # Online 2: dynamic Top-3 context must not silently choose the first SKU.
    top_three = run_agent("华东即时零售成交销额最高的3个SKU。")
    _assert_ready(top_three)
    product_context = top_three.turn_context
    product_ids = [item["product_id"] for item in product_context["entities"]["products"]]
    assert len(product_ids) == 3
    ambiguous = run_agent("它的毛利率呢？", conversation_context=product_context)
    assert ambiguous.plan.status == "needs_clarification" and ambiguous.plan.sql is None
    assert all(product_id in ambiguous.plan.clarification_question for product_id in product_ids)
    print(f"Online 2: PASS | clarification candidates={product_ids}")

    # Online 3: plural store reference inherits the complete result set.
    high_refund = run_agent("找出2025年上半年退损率超过5%的门店。")
    _assert_ready(high_refund)
    refund_context = high_refund.turn_context
    store_ids = [item["store_id"] for item in refund_context["entities"]["stores"]]
    assert set(store_ids) == {"S003", "S001"}
    plural = run_agent("它们Q2成交销额分别是多少？", conversation_context=refund_context)
    _assert_ready(plural)
    assert plural.conversation_context_used
    assert all(store_id in plural.plan.sql for store_id in store_ids)
    assert all(date in plural.plan.sql for date in ("2025-04-01", "2025-07-01"))
    assert set(plural.query_result.dataframe["store_id"].astype(str)) == set(store_ids)
    print(f"Online 3: PASS | inherited stores={store_ids}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    run_offline_tests()
    if args.online:
        run_online_tests()
    print("All requested conversation tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
