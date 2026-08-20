"""Deterministic multi-task planning, execution, chart, UI, and isolation tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from bupt_data_agent.agent import (
    AnalysisPlan,
    AnalysisTask,
    LLMConfig,
    SQLPlan,
    SemanticPlan,
    VisualizationSpec,
    _parse_analysis_plan,
    run_agent,
)
from bupt_data_agent.evidence import build_analysis_evidence


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "bupt_data_agent" / "streamlit_app.py"

REFUND_REASONS_SQL = (
    "SELECT DISTINCT refund_reason FROM refund_record ORDER BY refund_reason"
)
STORE_COUNT_SQL = "SELECT COUNT(*) AS store_count FROM store_info"
REFUND_COMPARISON_SQL = """
SELECT
    refund_reason,
    COUNT(*) AS refund_quantity,
    ROUND(SUM(refund_amount), 2) AS refund_amount
FROM refund_record
WHERE refund_reason IN ('质量问题', '运输破损')
GROUP BY refund_reason
ORDER BY refund_reason
"""
CATEGORY_AMOUNT_SQL = """
SELECT
    p.category,
    ROUND(SUM(s.quantity * s.sale_price - s.discount_amount), 2) AS total_amount
FROM sales_order AS s
JOIN product_info AS p ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY total_amount DESC, p.category
"""
TOP_PRODUCTS_SQL = """
SELECT
    p.product_id,
    p.product_name,
    ROUND(SUM(s.quantity * s.sale_price - s.discount_amount), 2) AS sales_amount
FROM sales_order AS s
JOIN product_info AS p ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name
ORDER BY sales_amount DESC, p.product_id
LIMIT 5
"""


def _semantic(
    *,
    intent: str,
    metrics: tuple[str, ...],
    dimensions: tuple[str, ...],
    tables: tuple[str, ...],
    chart: str = "none",
) -> SemanticPlan:
    return SemanticPlan(
        intent=intent,
        metrics=metrics,
        dimensions=dimensions,
        tables=tables,
        visualization_intent=chart,
    )


def _task(
    task_id: str,
    question: str,
    sql: str,
    semantic_plan: SemanticPlan,
    visualization: VisualizationSpec | None = None,
) -> AnalysisTask:
    return AnalysisTask(
        task_id=task_id,
        question=question,
        sql=sql.strip(),
        reasoning_summary=f"执行 {task_id} 的独立聚合查询。",
        visualization=visualization or VisualizationSpec(),
        semantic_plan=semantic_plan,
    )


def _run_with_plan(plan: AnalysisPlan):
    with (
        patch(
            "bupt_data_agent.agent.load_config",
            return_value=LLMConfig("hidden", "test", None),
        ),
        patch("bupt_data_agent.agent.create_llm_client", return_value=object()),
        patch("bupt_data_agent.agent.load_business_context", return_value="business"),
        patch("bupt_data_agent.agent.load_schema_context", return_value="schema"),
        patch("bupt_data_agent.agent.generate_sql_plan", return_value=plan),
        patch(
            "bupt_data_agent.agent.summarize_analysis_results",
            return_value="所有任务的统一测试结论。",
        ) as summarize,
    ):
        result = run_agent(plan.original_question)
    return result, summarize


def test_two_independent_sql_without_charts() -> None:
    question = "目前有哪几种退款原因，以及有几个店铺？"
    plan = AnalysisPlan(
        original_question=question,
        tasks=(
            _task(
                "task_1",
                "查询目前有哪些退款原因",
                REFUND_REASONS_SQL,
                _semantic(
                    intent="detail_query",
                    metrics=(),
                    dimensions=("退款原因",),
                    tables=("refund_record",),
                ),
            ),
            _task(
                "task_2",
                "查询店铺数量",
                STORE_COUNT_SQL,
                _semantic(
                    intent="aggregation",
                    metrics=("店铺数量",),
                    dimensions=(),
                    tables=("store_info",),
                ),
            ),
        ),
    )
    result, summarize = _run_with_plan(plan)
    assert len(result.analysis_plan.tasks) == 2
    assert len(result.task_results) == 2
    assert all(item.status == "success" for item in result.task_results)
    assert sum(item.query_result is not None for item in result.task_results) == 2
    assert all(item.chart_path is None for item in result.task_results)
    assert result.task_results[0].query_result.dataframe["refund_reason"].nunique() >= 2
    assert result.task_results[1].query_result.dataframe.iloc[0]["store_count"] == 4
    assert summarize.call_count == 1

    evidence = build_analysis_evidence(
        {
            "task_id": item.task.task_id,
            "question": item.task.question,
            "sql": item.task.sql,
            "business_validation": item.business_validation,
        }
        for item in result.task_results
    )
    assert [item["task_id"] for item in evidence] == ["task_1", "task_2"]
    assert evidence[0]["evidence"]["tables"] == ["refund_record"]
    assert evidence[1]["evidence"]["tables"] == ["store_info"]
    print("Test 1 two independent SQL/no charts/evidence isolation: PASS")


def test_two_tasks_two_charts_and_streamlit() -> None:
    question = (
        "分别统计因为质量问题、运输破损两个原因，用户退货的数量和总金额，"
        "并用柱状图做对比，并同时计算各商品类别（3C配件等）的总价格，画出柱状图。"
    )
    plan = AnalysisPlan(
        original_question=question,
        tasks=(
            _task(
                "task_1",
                "分别统计质量问题和运输破损的退货数量和总金额，并画柱状图",
                REFUND_COMPARISON_SQL,
                _semantic(
                    intent="comparison",
                    metrics=("退货数量", "退款总金额"),
                    dimensions=("退款原因",),
                    tables=("refund_record",),
                    chart="bar",
                ),
                VisualizationSpec(
                    required=True,
                    chart_type="bar",
                    x="refund_reason",
                    y=("refund_quantity", "refund_amount"),
                    title="不同退款原因的退货数量与金额对比",
                ),
            ),
            _task(
                "task_2",
                "统计各商品类别的总价格并画柱状图",
                CATEGORY_AMOUNT_SQL,
                _semantic(
                    intent="aggregation",
                    metrics=("成交销额",),
                    dimensions=("商品类别",),
                    tables=("sales_order", "product_info"),
                    chart="bar",
                ),
                VisualizationSpec(
                    required=True,
                    chart_type="bar",
                    x="category",
                    y=("total_amount",),
                    title="各商品类别总价格",
                ),
            ),
        ),
    )

    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "charts"
        with patch("bupt_data_agent.agent.OUTPUTS_DIR", output_dir):
            result, summarize = _run_with_plan(plan)
        chart_paths = [item.chart_path for item in result.task_results]
        assert len(result.task_results) == 2
        assert all(item.status == "success" for item in result.task_results)
        assert len([path for path in chart_paths if path and path.is_file()]) == 2
        assert len(set(chart_paths)) == 2
        assert result.task_results[0].evidence["filters"] == [
            "refund_reason IN ('质量问题', '运输破损')"
        ]
        assert "按商品类别聚合（category）" in (
            result.task_results[1].evidence["aggregation"]
        )
        assert summarize.call_count == 1

        history_db = Path(temporary_directory) / "chat_history.db"
        with (
            patch("bupt_data_agent.chat_history.CHAT_HISTORY_DB_PATH", history_db),
            patch("bupt_data_agent.agent.run_agent", return_value=result),
            patch("streamlit.image") as show_image,
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
            app.chat_input[0].set_value(question).run()
            show_image.reset_mock()
            app.run()
            assert not app.exception
            assert show_image.call_count == 2
            assert len(app.subheader) >= 2
            assert "分析任务 1" in app.subheader[0].value
            assert "分析任务 2" in app.subheader[1].value
            assert len(app.expander) == 2
            assert len(app.code) == 2
            assert "refund_record" in app.code[0].value
            assert "product_info" in app.code[1].value
        print("Test 2 two successful SQL/two saved charts/Streamlit images: PASS")


def test_single_task_compatibility() -> None:
    question = "查询销售额最高的 5 个商品。"
    plan = AnalysisPlan(
        original_question=question,
        tasks=(
            _task(
                "task_1",
                question,
                TOP_PRODUCTS_SQL,
                _semantic(
                    intent="ranking",
                    metrics=("成交销额",),
                    dimensions=("商品",),
                    tables=("sales_order", "product_info"),
                ),
            ),
        ),
    )
    result, _ = _run_with_plan(plan)
    assert len(result.task_results) == 1
    assert result.task_results[0].status == "success"
    assert len(result.task_results[0].query_result.dataframe) == 5
    assert result.plan.sql == TOP_PRODUCTS_SQL.strip()
    assert result.query_result is result.task_results[0].query_result
    print("Test 3 single-task compatibility as N=1: PASS")


def test_multiple_metrics_remain_one_task() -> None:
    question = "分别统计质量问题和运输破损的退款数量和退款金额。"
    payload = {
        "status": "ready",
        "clarification_question": None,
        "ambiguity_type": None,
        "original_question": question,
        "tasks": [
            {
                "task_id": "task_1",
                "question": question,
                "semantic_plan": {
                    "intent": "comparison",
                    "metrics": ["退款数量", "退款金额"],
                    "dimensions": ["退款原因"],
                    "time_range": None,
                    "filters": ["退款原因 IN (质量问题, 运输破损)"],
                    "tables": ["refund_record"],
                    "visualization_intent": "none",
                },
                "sql": REFUND_COMPARISON_SQL.strip(),
                "reasoning_summary": "同一退款原因维度一次返回两个指标。",
                "visualization": {
                    "required": False,
                    "chart_type": "none",
                    "x": None,
                    "y": [],
                    "title": None,
                },
            }
        ],
    }
    plan = _parse_analysis_plan(payload)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].semantic_plan.metrics == ("退款数量", "退款金额")
    print("Test 4 same grouping/multiple metrics remains one task: PASS")


def test_partial_failure_isolation() -> None:
    question = "查询店铺数量，并执行另一个测试分析。"
    broken_sql = "SELECT missing_column FROM store_info"
    plan = AnalysisPlan(
        original_question=question,
        tasks=(
            _task(
                "task_1",
                "查询店铺数量",
                STORE_COUNT_SQL,
                _semantic(
                    intent="aggregation",
                    metrics=("店铺数量",),
                    dimensions=(),
                    tables=("store_info",),
                ),
            ),
            _task(
                "task_2",
                "执行会失败的测试分析",
                broken_sql,
                _semantic(
                    intent="detail_query",
                    metrics=(),
                    dimensions=("测试字段",),
                    tables=("store_info",),
                ),
            ),
        ),
    )
    with patch(
        "bupt_data_agent.agent.repair_sql_plan",
        return_value=SQLPlan(broken_sql, "修复后仍然无效。", "none"),
    ):
        result, summarize = _run_with_plan(plan)
    assert [item.status for item in result.task_results] == ["success", "failed"]
    assert result.task_results[0].query_result is not None
    assert result.task_results[0].query_result.dataframe.iloc[0]["store_count"] == 4
    assert result.task_results[1].query_result is None
    assert result.task_results[1].error_type == "SQLExecutionError"
    assert result.task_results[1].error_message
    assert summarize.call_count == 1
    print("Test 5 partial failure preserves successful task: PASS")


def main() -> None:
    test_two_independent_sql_without_charts()
    test_two_tasks_two_charts_and_streamlit()
    test_single_task_compatibility()
    test_multiple_metrics_remain_one_task()
    test_partial_failure_isolation()
    print("All deterministic multi-task tests passed.")


if __name__ == "__main__":
    main()
