"""Deterministic persistence checks for the local chat-history database."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

import bupt_data_agent.chat_history as chat_history


def main() -> None:
    original_path = chat_history.CHAT_HISTORY_DB_PATH
    try:
        with TemporaryDirectory() as temporary_directory:
            test_db = Path(temporary_directory) / "chat_history.db"
            chat_history.CHAT_HISTORY_DB_PATH = test_db
            chat_history.initialize_chat_history()
            assert test_db.is_file()

            first_id = chat_history.create_conversation("第一段分析")
            chat_history.append_message(first_id, "user", "查询门店销售额")
            assistant_payload = {
                "kind": "result",
                "details": {
                    "semantic_plan": {
                        "intent": "ranking",
                        "metrics": ["成交销额", "销量"],
                        "dimensions": ["SKU", "品类"],
                        "time_range": "2025-H1",
                        "filters": ["战区=华东", "渠道=O2O", "Top3"],
                        "tables": ["sales_order", "store_info", "product_info"],
                        "visualization_intent": "none",
                    },
                    "sql": "SELECT store_id FROM sales_order",
                    "evidence": {"tables": ["sales_order"]},
                    "query_data": {
                        "columns": ["store_id", "sales_amount"],
                        "records": [{"store_id": "S001", "sales_amount": 1.0}],
                    },
                },
            }
            chat_history.append_message(
                first_id,
                "assistant",
                "S001销售额最高。",
                assistant_payload,
            )
            context = {
                "entities": {"stores": [{"store_id": "S001"}], "products": []},
                "time_context": {"year": 2025, "period": "H1"},
            }
            chat_history.update_conversation_context(first_id, context)
            semantic_title = chat_history.semantic_conversation_title(
                {
                    "intent": "ranking",
                    "metrics": ["成交销额", "销量"],
                    "dimensions": ["SKU", "品类"],
                    "time_range": "2025-H1",
                    "filters": ["战区=华东", "渠道=O2O", "Top3"],
                    "tables": ["sales_order", "store_info", "product_info"],
                    "visualization_intent": "none",
                }
            )
            assert semantic_title == "华东 O2O Top3 SKU"
            chat_history.update_conversation_title(first_id, semantic_title)

            second_id = chat_history.create_conversation("第二段分析")
            chat_history.append_message(second_id, "user", "查询SKU")
            assert chat_history.list_conversations()[0]["conversation_id"] == second_id

            chat_history.update_conversation_context(first_id, context)
            assert chat_history.list_conversations()[0]["conversation_id"] == first_id

            restored = chat_history.load_conversation(first_id)
            assert restored is not None
            assert restored["title"] == "华东 O2O Top3 SKU"
            assert restored["last_turn_context"] == context
            assert [message["role"] for message in restored["messages"]] == [
                "user",
                "assistant",
            ]
            restored_payload = restored["messages"][1]["payload"]
            assert restored_payload["details"]["semantic_plan"]["intent"] == "ranking"
            assert restored_payload["details"]["query_data"]["records"][0][
                "store_id"
            ] == "S001"
            print("Chat history create/append/payload/context/reload/order: PASS")

            assert chat_history.conversation_title(
                "查询2025年上半年每家门店的销售额，从高到低排序。"
            ) == "2025H1 门店销额排名"
            assert chat_history.semantic_conversation_title(
                {
                    "intent": "ranking",
                    "metrics": ["成交销额"],
                    "dimensions": ["门店"],
                    "time_range": "2025-H1",
                    "filters": [],
                }
            ) == "2025H1 门店销额排名"
            assert chat_history.semantic_conversation_title(
                {
                    "intent": "drill_down",
                    "metrics": ["成交销额增长率", "毛利率"],
                    "dimensions": ["门店", "SKU"],
                    "time_range": "2025-Q1 vs 2025-Q2",
                    "filters": ["增长率>10%", "Q2毛利率<Q1毛利率"],
                }
            ) == "Q2增长与毛利下钻"
            print("Fallback and SemanticPlan conversation titles: PASS")

            legacy_id = chat_history.create_conversation(
                "比较每家门店2025年第一季度和第二季度销售额……"
            )
            chat_history.append_message(
                legacy_id,
                "user",
                "比较每家门店2025年第一季度和第二季度销售额，找出增长门店。",
            )
            chat_history.append_message(
                legacy_id,
                "assistant",
                "查询完成。",
                {
                    "kind": "result",
                    "details": {
                        "semantic_plan": {
                            "intent": "comparison",
                            "metrics": ["成交销额", "季度环比增长率"],
                            "dimensions": ["门店"],
                            "time_range": "2025年第一季度至第二季度（2025-01-01至2025-07-01）",
                            "filters": ["第二季度销售额较第一季度增长超过10%"],
                        }
                    },
                },
            )
            fallback_id = chat_history.create_conversation(
                "查询2025年上半年每家门店销售额，从高到低……"
            )
            chat_history.append_message(
                fallback_id,
                "user",
                "查询2025年上半年每家门店销售额，从高到低排序。",
            )
            chat_history.initialize_chat_history()
            assert chat_history.load_conversation(legacy_id)["title"] == (
                "Q1/Q2 门店销售对比"
            )
            assert chat_history.load_conversation(fallback_id)["title"] == (
                "2025H1 门店销额排名"
            )
            print("Existing conversation title migration: PASS")

            chat_history.delete_conversation(second_id)
            assert chat_history.load_conversation(second_id) is None
            print("Chat history delete: PASS")
            print("All chat history tests passed.")
    finally:
        chat_history.CHAT_HISTORY_DB_PATH = original_path


if __name__ == "__main__":
    main()
