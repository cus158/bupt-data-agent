"""Offline Streamlit AppTest for submission, persistence, and clarification flows."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from bupt_data_agent.agent import AgentResult, QueryResult, SQLPlan, SemanticPlan
from bupt_data_agent.business_validator import BusinessValidationResult


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "bupt_data_agent" / "streamlit_app.py"
EXPECTED_EXAMPLE_LABELS = (
    "门店销额",
    "华东 O2O Top3 SKU",
    "高退损门店",
    "季度增长",
    "毛利下降下钻",
)


def _fake_result(question: str, conversation_context=None) -> AgentResult:
    if "表现最好" in question:
        plan = SQLPlan(
            sql=None,
            reasoning_summary="指标存在歧义。",
            chart_type="none",
            status="needs_clarification",
            clarification_question="表现最好可以按成交销额、销量或毛利率衡量，请明确指标。",
            ambiguity_type="metric",
        )
        return AgentResult(
            question=question,
            first_plan=plan,
            plan=plan,
            query_result=None,
            conclusion=None,
            repair_triggered=False,
            first_error_type=None,
            first_error_message=None,
            business_validation=None,
            turn_context=None,
            conversation_context_used=False,
        )

    plan = SQLPlan(
        sql=(
            "SELECT s.product_id, p.product_name, p.category, "
            "SUM(s.quantity) AS sales_qty, "
            "SUM(s.quantity * s.sale_price - s.discount_amount) AS sales_amount "
            "FROM sales_order s JOIN store_info st ON st.store_id=s.store_id "
            "JOIN product_info p ON p.product_id=s.product_id "
            "WHERE s.order_date >= '2025-01-01' AND s.order_date < '2025-07-01' "
            "AND st.region='华东' AND s.channel_code='O2O' "
            "GROUP BY s.product_id, p.product_name, p.category "
            "ORDER BY sales_amount DESC LIMIT 3"
        ),
        reasoning_summary="按华东O2O成交销额查询Top3 SKU。",
        chart_type="none",
        semantic_plan=SemanticPlan(
            intent="ranking",
            metrics=("成交销额", "销量"),
            dimensions=("SKU", "品类"),
            time_range="2025-H1",
            filters=("战区=华东", "渠道=O2O", "Top3"),
            tables=("sales_order", "store_info", "product_info"),
            visualization_intent="none",
        ),
    )
    dataframe = pd.DataFrame(
        [
            {"product_id": "P008", "product_name": "24寸显示器", "sales_qty": 134, "sales_amount": 101287.91, "category": "办公设备"},
            {"product_id": "P007", "product_name": "激光打印机", "sales_qty": 102, "sales_amount": 97260.10, "category": "办公设备"},
            {"product_id": "P001", "product_name": "商务耳机", "sales_qty": 268, "sales_amount": 75544.43, "category": "3C配件"},
        ]
    )
    turn_context = {
        "question": question,
        "entities": {
            "stores": [],
            "products": [
                {"product_id": "P008", "product_name": "24寸显示器"},
                {"product_id": "P007", "product_name": "激光打印机"},
                {"product_id": "P001", "product_name": "商务耳机"},
            ],
        },
        "entity_counts": {"stores": 0, "products": 3},
        "time_context": {"year": 2025, "period": "H1"},
        "metrics": ["成交销额", "销量"],
        "result_summary": "SKU：P008、P007、P001",
    }
    return AgentResult(
        question=question,
        first_plan=plan,
        plan=plan,
        query_result=QueryResult(dataframe, False),
        conclusion="P008、P007、P001 为华东即时零售成交销额最高的3个SKU。",
        repair_triggered=False,
        first_error_type=None,
        first_error_message=None,
        business_validation=BusinessValidationResult(True, (), ()),
        turn_context=turn_context,
        conversation_context_used=conversation_context is not None,
    )


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def main() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")
    assert ".st-key-composer" in app_source
    assert "position: fixed" in app_source
    assert "padding-bottom: 9rem" in app_source
    assert "--agent-sidebar-width" in app_source
    assert "flex: 0 0 auto !important" in app_source
    assert "linear-gradient(" in app_source
    assert "stChatInputSubmitButton" in app_source
    print("UI fixed floating Composer/chips/input CSS: PASS")

    with TemporaryDirectory() as temporary_directory:
        history_db = Path(temporary_directory) / "chat_history.db"
        with (
            patch("bupt_data_agent.chat_history.CHAT_HISTORY_DB_PATH", history_db),
            patch("bupt_data_agent.agent.run_agent", side_effect=_fake_result) as run_agent,
            patch("bupt_data_agent.agent.create_chart") as create_chart,
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
            assert not app.exception
            assert run_agent.call_count == 0
            labels = {button.label for button in app.button}
            assert "＋ 开启新对话" in labels
            assert set(EXPECTED_EXAMPLE_LABELS).issubset(labels)
            assert len(app.chat_input) == 1
            assert any("用自然语言查询门店经营数据" in item.value for item in app.markdown)
            print("UI Scenario 1 initial/no API/examples/chat input: PASS")

            _button(app, "华东 O2O Top3 SKU").click().run()
            app.run()
            assert not app.exception
            assert run_agent.call_count == 1
            assert create_chart.call_count == 0
            assert app.session_state["last_turn_context"]["entity_counts"]["products"] == 3
            assert any(expander.label == "问题理解与执行过程" for expander in app.expander)
            assert any("ranking" in markdown.value for markdown in app.markdown)
            assert len(app.chat_input) == 1
            assert "华东 O2O Top3 SKU" in {button.label for button in app.button}
            print("UI Scenario 2/3 Q2 single submit and SemanticPlan: PASS")

            app.run()
            assert run_agent.call_count == 1
            print("UI Scenario 4 ordinary rerun no Agent call: PASS")

            saved_context = app.session_state["last_turn_context"]
            _button(app, "＋ 开启新对话").click().run()
            assert run_agent.call_count == 1
            assert app.session_state["messages"] == []
            assert app.session_state["last_turn_context"] is None
            history_button = next(
                button
                for button in app.button
                if button.label == "华东 O2O Top3 SKU"
                and str(button.key).startswith("history_")
            )
            history_button.click().run()
            assert run_agent.call_count == 1
            assert len(app.session_state["messages"]) == 2
            assert app.session_state["last_turn_context"] == saved_context
            assert len(app.chat_input) == 1
            assert set(EXPECTED_EXAMPLE_LABELS).issubset(
                {button.label for button in app.button}
            )
            print("UI Scenario 5/6/7 history switch, context restore, new dialog: PASS")

            app.chat_input[0].set_value("这些SKU的销量再比较一下").run()
            app.run()
            assert run_agent.call_count == 2
            assert len(app.session_state["messages"]) == 4
            assert len(app.chat_input) == 1
            assert set(EXPECTED_EXAMPLE_LABELS).issubset(
                {button.label for button in app.button}
            )
            print("UI multi-turn composer remains below latest messages: PASS")

            _button(app, "＋ 开启新对话").click().run()
            app.chat_input[0].set_value("哪个商品表现最好？").run()
            assert run_agent.call_count == 3
            assert app.session_state["messages"][-1]["kind"] == "clarification"
            assert not app.expander
            print("UI Scenario 9 clarification without fake execution details: PASS")

            assert create_chart.call_count == 0
            app.run()
            assert create_chart.call_count == 0
            print("UI Scenario 10 history/rerun never regenerates charts: PASS")
            print("All Streamlit AppTest scenarios passed.")


if __name__ == "__main__":
    main()
