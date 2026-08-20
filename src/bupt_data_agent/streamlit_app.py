"""Conversational Streamlit presentation layer for the Text-to-SQL agent."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from bupt_data_agent.agent import (
    AgentError,
    BusinessRuleValidationError,
    ConfigurationError,
    LLMResponseError,
    MAX_RESULT_ROWS,
    SQLExecutionError,
    SQLSafetyError,
    TaskResult,
    create_chart,
    run_agent,
)
from bupt_data_agent.chat_history import (
    append_message,
    conversation_title,
    create_conversation,
    initialize_chat_history,
    list_conversations,
    load_conversation,
    semantic_conversation_title,
    update_conversation_context,
    update_conversation_title,
)
from bupt_data_agent.conversation import context_display_summary
from bupt_data_agent.evidence import build_query_evidence
from bupt_data_agent.paths import DB_PATH, ENV_FILE


EXAMPLE_QUESTIONS = {
    "门店销额": "查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    "华东 O2O Top3 SKU": "查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。",
    "高退损门店": "查询各门店 2025 年上半年的退损情况，找出退损率超过 5% 的门店，画出各门店退损率对比图，并分析退损率较高门店的主要退款原因。",
    "季度增长": "比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。",
    "毛利下降下钻": "找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。",
}


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 960px;
            padding-top: 1rem;
            padding-bottom: 9rem;
        }
        .block-container h1 {
            font-size: 1.65rem;
            margin-bottom: 0.1rem;
        }
        div[data-testid="stChatMessage"] {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            gap: 0.65rem;
        }
        div[data-testid="stChatMessageAvatarUser"],
        div[data-testid="stChatMessageAvatarAssistant"] {
            width: 1.8rem;
            height: 1.8rem;
        }
        section[data-testid="stSidebar"] {
            width: 270px !important;
            min-width: 270px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 1rem 0.75rem;
        }
        section[data-testid="stSidebar"] h1 {
            font-size: 1.15rem;
            margin-bottom: 0.55rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            min-height: 2rem;
            padding: 0.25rem 0.55rem;
            border: 0;
            border-radius: 0.4rem;
            background: transparent;
            justify-content: flex-start;
            font-size: 0.86rem;
            font-weight: 400;
        }
        section[data-testid="stSidebar"] [class*="st-key-history_"] button {
            overflow: hidden;
        }
        section[data-testid="stSidebar"] [class*="st-key-history_"] button p {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            text-align: left;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background: #f3f5f8;
            color: inherit;
        }
        section[data-testid="stSidebar"] [class*="st-key-history_"] button:disabled {
            color: #1f5fae;
            background: #eaf2fc;
            opacity: 1;
        }
        section[data-testid="stSidebar"] .st-key-new_conversation button {
            border: 1px solid #b9d3f3 !important;
            background: #f5f9ff !important;
            color: #155ca6 !important;
            font-weight: 500 !important;
            justify-content: center !important;
        }
        .st-key-system_status {
            color: #667085;
            font-size: 0.78rem;
        }
        .st-key-empty_state {
            text-align: center;
            color: #667085;
            padding: 18vh 0 1.5rem;
        }
        .st-key-empty_state h3 {
            color: #344054;
            font-size: 1.1rem;
            font-weight: 500;
        }
        body {
            --agent-sidebar-width: 0px;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) {
            --agent-sidebar-width: 270px;
        }
        .st-key-composer {
            position: fixed;
            bottom: 14px;
            left: calc(
                var(--agent-sidebar-width)
                + (100vw - var(--agent-sidebar-width)) / 2
            );
            transform: translateX(-50%);
            width: min(
                960px,
                calc(100vw - var(--agent-sidebar-width) - 2rem)
            );
            z-index: 20;
            margin: 0;
            padding: 1.15rem 0 0;
            background: linear-gradient(
                to bottom,
                rgba(255, 255, 255, 0),
                rgba(255, 255, 255, 0.94) 32%,
                rgba(255, 255, 255, 1) 58%
            );
        }
        .st-key-composer [data-testid="stHorizontalBlock"] {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.45rem;
            margin-bottom: 0.45rem;
        }
        .st-key-composer [data-testid="stColumn"] {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        .st-key-composer div[data-testid="stButton"] button {
            width: auto !important;
            min-height: 2.1rem;
            height: 2.1rem;
            padding: 0 0.9rem;
            border: 1px solid #e0e3e8;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.96);
            color: #344054;
            font-size: 0.875rem;
            font-weight: 400;
            white-space: nowrap;
            box-shadow: none;
        }
        .st-key-composer div[data-testid="stButton"] button:hover {
            border-color: #d2d6dc;
            background: #f5f5f5;
            color: #344054;
        }
        .st-key-composer [data-testid="stChatInput"] {
            min-height: 3.2rem;
            border: 0 !important;
            border-radius: 1.65rem;
            background: #f4f4f4;
            box-shadow: none;
        }
        .st-key-composer [data-baseweb="textarea"] {
            border: 0 !important;
            border-radius: 1.65rem;
            background: #f4f4f4;
            box-shadow: none !important;
        }
        .st-key-composer [data-testid="stChatInput"] textarea {
            color: #202124;
            background: transparent;
        }
        .st-key-composer [data-testid="stChatInput"] textarea::placeholder {
            color: #98a0aa;
        }
        .st-key-composer [data-testid="stChatInputSubmitButton"] {
            border-radius: 999px;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-bottom: 12rem;
            }
            .st-key-composer {
                width: calc(100vw - var(--agent-sidebar-width) - 1rem);
                bottom: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configured_model_name() -> str:
    load_dotenv(ENV_FILE, override=False)
    return os.getenv("LLM_MODEL", "").strip() or "未配置"


def llm_configuration_status() -> str:
    load_dotenv(ENV_FILE, override=False)
    return (
        "Ready"
        if os.getenv("LLM_API_KEY", "").strip()
        and os.getenv("LLM_MODEL", "").strip()
        else "Not configured"
    )


def initialize_session_state() -> None:
    initialize_chat_history()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_conversation_id", None)
    st.session_state.setdefault("last_turn_context", None)


def _analysis_error_message(exc: Exception) -> str:
    if isinstance(exc, ConfigurationError):
        return "模型配置不可用，请检查 .env。"
    if isinstance(exc, SQLSafetyError):
        return "生成的 SQL 未通过安全校验。"
    if isinstance(exc, BusinessRuleValidationError):
        messages = "；".join(issue.message for issue in exc.result.violations)
        return f"生成的查询未通过业务规则校验：{messages}"
    if isinstance(exc, SQLExecutionError):
        return "查询执行失败，请检查问题后重试。"
    if isinstance(exc, LLMResponseError):
        return "模型返回格式异常，请稍后重试。"
    if isinstance(exc, openai.AuthenticationError):
        return "模型服务认证失败，请检查 .env 配置。"
    if isinstance(exc, openai.APITimeoutError):
        return "模型服务请求超时，请稍后重试。"
    if isinstance(exc, openai.APIConnectionError):
        return "模型服务请求失败，请检查网络后重试。"
    if isinstance(exc, openai.APIStatusError):
        return f"模型服务请求失败（HTTP {exc.status_code}），请稍后重试。"
    if isinstance(exc, AgentError):
        return "分析执行失败，请稍后重试。"
    return "分析过程中发生未预期错误。"


def _dataframe_payload(dataframe: pd.DataFrame) -> dict[str, Any] | None:
    try:
        records = json.loads(
            dataframe.to_json(
                orient="records", date_format="iso", force_ascii=False
            )
        )
        return {
            "columns": [str(column) for column in dataframe.columns],
            "records": records,
        }
    except (TypeError, ValueError, OverflowError):
        logging.exception("Query data could not be serialized for chat history")
        return None


def _assistant_payload(message: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": message.get("kind", "result"),
        "ambiguity_type": message.get("ambiguity_type"),
    }
    details = message.get("details")
    if isinstance(details, dict):
        serializable_details = {
            key: value
            for key, value in details.items()
            if key not in {"dataframe", "tasks"}
        }
        dataframe = details.get("dataframe")
        if isinstance(dataframe, pd.DataFrame):
            serializable_details["query_data"] = _dataframe_payload(dataframe)
        task_details = details.get("tasks")
        if isinstance(task_details, list):
            serializable_details["tasks"] = []
            for task in task_details:
                if not isinstance(task, dict):
                    continue
                serializable_task = {
                    key: value for key, value in task.items() if key != "dataframe"
                }
                task_dataframe = task.get("dataframe")
                if isinstance(task_dataframe, pd.DataFrame):
                    serializable_task["query_data"] = _dataframe_payload(task_dataframe)
                serializable_details["tasks"].append(serializable_task)
        payload["details"] = serializable_details
    if message.get("semantic_plan") is not None:
        payload["semantic_plan"] = message["semantic_plan"]
    return payload


def _restore_message(stored_message: dict[str, Any]) -> dict[str, Any]:
    message = {
        "role": stored_message["role"],
        "content": stored_message["content"],
    }
    payload = stored_message.get("payload")
    if not isinstance(payload, dict):
        return message
    message.update(
        {
            "kind": payload.get("kind", "result"),
            "ambiguity_type": payload.get("ambiguity_type"),
        }
    )
    if payload.get("semantic_plan") is not None:
        message["semantic_plan"] = payload["semantic_plan"]
    details = payload.get("details")
    if isinstance(details, dict):
        details = details.copy()
        query_data = details.pop("query_data", None)
        dataframe: pd.DataFrame | None = None
        if isinstance(query_data, dict):
            columns = query_data.get("columns")
            records = query_data.get("records")
            if isinstance(columns, list) and isinstance(records, list):
                dataframe = pd.DataFrame(records, columns=columns)
        details["dataframe"] = dataframe
        details["query_data_saved"] = dataframe is not None
        restored_tasks: list[dict[str, Any]] = []
        for task in details.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            restored_task = task.copy()
            task_query_data = restored_task.pop("query_data", None)
            task_dataframe: pd.DataFrame | None = None
            if isinstance(task_query_data, dict):
                task_columns = task_query_data.get("columns")
                task_records = task_query_data.get("records")
                if isinstance(task_columns, list) and isinstance(task_records, list):
                    task_dataframe = pd.DataFrame(task_records, columns=task_columns)
            restored_task["dataframe"] = task_dataframe
            restored_task["query_data_saved"] = task_dataframe is not None
            restored_tasks.append(restored_task)
        if "tasks" in details:
            details["tasks"] = restored_tasks
        message["details"] = details
    return message


def start_new_conversation() -> None:
    st.session_state["active_conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["last_turn_context"] = None


def load_conversation_into_session(conversation_id: str) -> None:
    conversation = load_conversation(conversation_id)
    if conversation is None:
        return
    st.session_state["active_conversation_id"] = conversation_id
    st.session_state["messages"] = [
        _restore_message(message) for message in conversation["messages"]
    ]
    st.session_state["last_turn_context"] = conversation["last_turn_context"]


def _history_group(updated_at: str) -> str:
    try:
        updated_date = datetime.fromisoformat(updated_at).astimezone().date()
    except (TypeError, ValueError):
        return "更早"
    today = datetime.now().astimezone().date()
    days = (today - updated_date).days
    if days == 0:
        return "今天"
    if days == 1:
        return "昨天"
    return "更早"


def show_sidebar() -> None:
    with st.sidebar:
        st.title("Data Analysis Agent")
        if st.button(
            "＋ 开启新对话", key="new_conversation", use_container_width=True
        ):
            start_new_conversation()
            st.rerun()

        conversations = list_conversations()
        active_id = st.session_state.get("active_conversation_id")
        for group in ("今天", "昨天", "更早"):
            group_items = [
                item for item in conversations if _history_group(item["updated_at"]) == group
            ]
            if not group_items:
                continue
            st.caption(group)
            for item in group_items:
                selected = item["conversation_id"] == active_id
                if st.button(
                    item["title"],
                    key=f"history_{item['conversation_id']}",
                    use_container_width=True,
                    disabled=selected,
                ):
                    load_conversation_into_session(item["conversation_id"])
                    st.rerun()
        with st.container(key="system_status"):
            st.divider()
            st.caption("System")
            labels, values = st.columns([0.9, 1.1])
            with labels:
                st.caption("DeepSeek")
                st.caption("SQLite")
                st.caption("Validator")
            with values:
                st.caption(llm_configuration_status())
                st.caption("Read-only" if DB_PATH.is_file() else "Unavailable")
                st.caption("Enabled")


def _validation_details(validation: Any) -> dict[str, Any]:
    return {
        "business_validation_valid": bool(validation and validation.valid),
        "business_warnings": (
            [warning.message for warning in validation.warnings]
            if validation
            else []
        ),
    }


def _task_result_details(task_result: TaskResult) -> dict[str, Any]:
    task = task_result.task
    query_result = task_result.query_result
    details: dict[str, Any] = {
        "task_id": task.task_id,
        "question": task.question,
        "status": task_result.status,
        "semantic_plan": asdict(task.semantic_plan) if task.semantic_plan else None,
        "sql": task.sql,
        "reasoning_summary": task.reasoning_summary,
        "visualization": asdict(task.visualization),
        "chart_type": task.chart_type,
        "chart_path": str(task_result.chart_path) if task_result.chart_path else None,
        "chart_error": (
            "图表生成失败，但不影响查询结果与分析结论。"
            if task_result.chart_error
            else None
        ),
        "evidence": task_result.evidence,
        "evidence_error": (
            "查询依据生成失败，但不影响实际查询结果。"
            if task_result.evidence_error
            else None
        ),
        "sql_safety_valid": task_result.status == "success",
        "repair_triggered": task_result.repair_triggered,
        "first_error_type": task_result.first_error_type,
        "error_type": task_result.error_type,
        "error_message": task_result.error_message,
        "dataframe": query_result.dataframe.copy() if query_result else None,
        "truncated": query_result.truncated if query_result else False,
    }
    details.update(_validation_details(task_result.business_validation))
    return details


def _legacy_task_details(
    question: str,
    result: Any,
    previous_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Adapt older mocked/saved AgentResult objects to one task."""
    if result.query_result is None or result.plan.sql is None:
        raise AgentError("Ready plan did not return an executable query result")
    chart_path: str | None = None
    chart_error: str | None = None
    if result.plan.chart_type != "none":
        try:
            generated_path = create_chart(
                result.query_result.dataframe,
                result.plan.chart_type,
                question,
            )
            chart_path = str(generated_path) if generated_path else None
        except Exception:
            logging.exception("Chart generation failed")
            chart_error = "图表生成失败，但不影响查询结果与分析结论。"

    evidence = None
    evidence_error = None
    try:
        evidence = build_query_evidence(
            result.plan.sql,
            question,
            result.business_validation,
        )
        if result.conversation_context_used:
            summary = context_display_summary(previous_context)
            evidence.setdefault("notes", []).insert(
                0,
                "对话上下文：本任务使用了上一轮结构化查询上下文"
                + (f"（{summary}）" if summary else "。"),
            )
    except Exception:
        logging.exception("Query evidence generation failed")
        evidence_error = "查询依据生成失败，但不影响实际查询结果。"

    semantic_plan = (
        asdict(result.plan.semantic_plan) if result.plan.semantic_plan else None
    )
    details = {
        "task_id": "task_1",
        "question": question,
        "status": "success",
        "semantic_plan": semantic_plan,
        "sql": result.plan.sql,
        "reasoning_summary": result.plan.reasoning_summary,
        "visualization": {
            "required": result.plan.chart_type != "none",
            "chart_type": result.plan.chart_type,
            "x": None,
            "y": [],
            "title": question if result.plan.chart_type != "none" else None,
        },
        "dataframe": result.query_result.dataframe.copy(),
        "truncated": result.query_result.truncated,
        "chart_type": result.plan.chart_type,
        "chart_path": chart_path,
        "chart_error": chart_error,
        "evidence": evidence,
        "evidence_error": evidence_error,
        "sql_safety_valid": True,
        "repair_triggered": result.repair_triggered,
        "first_error_type": result.first_error_type,
        "error_type": None,
        "error_message": None,
    }
    details.update(_validation_details(result.business_validation))
    return details


def analyze_question(question: str) -> dict[str, Any]:
    """Run one Agent turn and persist all task-scoped results and artifacts."""
    previous_context = st.session_state.get("last_turn_context")
    try:
        result = run_agent(question, conversation_context=previous_context)
        if result.plan.status == "needs_clarification":
            semantic_plan = (
                asdict(result.plan.semantic_plan) if result.plan.semantic_plan else None
            )
            return {
                "role": "assistant",
                "kind": "clarification",
                "content": result.plan.clarification_question or "请补充查询条件。",
                "ambiguity_type": result.plan.ambiguity_type,
                "semantic_plan": semantic_plan,
            }

        task_details = (
            [_task_result_details(item) for item in result.task_results]
            if result.task_results
            else [_legacy_task_details(question, result, previous_context)]
        )
        success_count = sum(item["status"] == "success" for item in task_details)
        first_task = task_details[0]
        details: dict[str, Any] = {
            "question": question,
            "tasks": task_details,
            "task_count": len(task_details),
            "success_count": success_count,
            "failed_count": len(task_details) - success_count,
            "semantic_plan": first_task.get("semantic_plan"),
            "conversation_context_used": result.conversation_context_used,
            "model": configured_model_name(),
        }
        # Keep the former top-level fields for one-task history and integrations.
        if len(task_details) == 1:
            details.update(first_task)
            details["tasks"] = task_details
            details["task_count"] = 1
            details["success_count"] = success_count
            details["failed_count"] = 1 - success_count

        st.session_state["last_turn_context"] = result.turn_context
        return {
            "role": "assistant",
            "kind": "result",
            "content": result.conclusion or "查询已完成。",
            "details": details,
        }
    except Exception as exc:
        logging.exception("Streamlit Agent turn failed")
        return {
            "role": "assistant",
            "kind": "error",
            "content": _analysis_error_message(exc),
        }


def submit_question(question: str) -> None:
    """Single submission path shared by chat input and official example buttons."""
    question = question.strip()
    if not question:
        return
    had_successful_result = any(
        message.get("role") == "assistant" and message.get("kind") == "result"
        for message in st.session_state["messages"]
    )
    conversation_id = st.session_state.get("active_conversation_id")
    if not conversation_id:
        conversation_id = create_conversation(conversation_title(question))
        st.session_state["active_conversation_id"] = conversation_id

    user_message = {"role": "user", "content": question}
    st.session_state["messages"].append(user_message)
    append_message(conversation_id, "user", question)

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Agent 正在分析数据……", expanded=True) as status:
            st.write(
                "执行流程：业务理解 → SQL生成 → 安全与业务规则校验 → "
                "数据库查询 → 结果分析"
            )
            assistant_message = analyze_question(question)
            if assistant_message["kind"] == "error":
                status.update(label="分析未完成", state="error", expanded=False)
            elif assistant_message["kind"] == "clarification":
                status.update(label="需要补充信息", state="complete", expanded=False)
            else:
                status.update(label="分析完成", state="complete", expanded=False)

    st.session_state["messages"].append(assistant_message)
    append_message(
        conversation_id,
        "assistant",
        assistant_message["content"],
        _assistant_payload(assistant_message),
    )
    if assistant_message["kind"] == "result":
        update_conversation_context(
            conversation_id, st.session_state.get("last_turn_context")
        )
        if not had_successful_result:
            details = assistant_message.get("details")
            semantic_plan = (
                details.get("semantic_plan") if isinstance(details, dict) else None
            )
            improved_title = semantic_conversation_title(semantic_plan)
            if improved_title:
                update_conversation_title(conversation_id, improved_title)
    st.rerun()


def _format_amount(value: float) -> str:
    return f"{value:,.2f}"


def _format_rate(value: float, column: str) -> str:
    percentage = value if "pct" in column.lower() or abs(value) > 1 else value * 100
    return f"{percentage:.2f}%"


def _styled_dataframe(dataframe: pd.DataFrame) -> pd.io.formats.style.Styler:
    formatters: dict[str, Any] = {}
    for column in dataframe.columns:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            continue
        name = str(column).lower()
        if any(token in name for token in ("rate", "ratio", "margin", "share", "pct")):
            formatters[column] = lambda value, column=name: (
                "—" if pd.isna(value) else _format_rate(float(value), column)
            )
        elif any(
            token in name
            for token in ("amount", "sales", "profit", "cost", "price", "discount")
        ):
            formatters[column] = lambda value: (
                "—" if pd.isna(value) else _format_amount(float(value))
            )
        elif any(token in name for token in ("quantity", "count", "number", "qty")):
            formatters[column] = lambda value: (
                "—" if pd.isna(value) else f"{float(value):,.0f}"
            )
    return dataframe.style.format(formatters, na_rep="—")


def show_query_data(details: dict[str, Any]) -> None:
    dataframe = details.get("dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        st.info("该历史消息未保存完整查询表格。")
        return
    if dataframe.empty:
        st.info("查询执行成功，但没有符合条件的数据。")
        return
    st.dataframe(
        _styled_dataframe(dataframe),
        use_container_width=True,
        hide_index=True,
        height=min(520, 90 + 35 * min(len(dataframe), 12)),
    )
    if details.get("truncated"):
        st.warning(f"结果较多，仅展示前 {MAX_RESULT_ROWS} 行。")


def _show_evidence_section(title: str, values: list[str], *, code: bool = False) -> None:
    if not values:
        return
    st.markdown(f"**{title}**")
    for value in values:
        st.markdown(f"- {'`' + value + '`' if code else value}")


def show_evidence(details: dict[str, Any]) -> None:
    evidence = details.get("evidence")
    if not evidence:
        st.warning(details.get("evidence_error") or "暂无可展示的查询依据。")
        return
    _show_evidence_section("使用的数据表", evidence.get("tables", []), code=True)
    _show_evidence_section("业务口径", evidence.get("business_terms", []))
    _show_evidence_section("时间范围", evidence.get("time_rules", []))
    _show_evidence_section(
        "筛选与聚合",
        evidence.get("filters", []) + evidence.get("aggregation", []),
    )
    for note in evidence.get("notes", []):
        st.info(note)
    for warning in evidence.get("business_warnings", []):
        st.warning(f"业务规则提醒：{warning}")
    if details.get("conversation_context_used"):
        st.caption("✓ Previous Context Used")


def show_semantic_plan(details: dict[str, Any]) -> None:
    plan = details.get("semantic_plan")
    if not isinstance(plan, dict):
        st.info("未提供结构化计划。")
        return
    fields = (
        ("分析意图", plan.get("intent")),
        ("分析指标", plan.get("metrics")),
        ("分析维度", plan.get("dimensions")),
        ("时间范围", plan.get("time_range")),
        ("筛选条件", plan.get("filters")),
        ("涉及数据表", plan.get("tables")),
        ("可视化需求", plan.get("visualization_intent")),
    )
    for label, value in fields:
        if value in (None, "", [], ()):
            continue
        display = (
            "、".join(str(item) for item in value)
            if isinstance(value, (list, tuple))
            else str(value)
        )
        st.markdown(f"**{label}**　{display}")


def show_execution_process(
    details: dict[str, Any],
    *,
    label: str = "问题理解与执行过程",
) -> None:
    with st.expander(label, expanded=False):
        st.markdown("#### ① 问题理解")
        show_semantic_plan(details)
        st.divider()
        st.markdown("#### ② SQL生成")
        st.caption(
            "SQL版本：修复后 SQL"
            if details.get("repair_triggered")
            else "SQL版本：首次生成 SQL"
        )
        st.code(details.get("sql") or "", language="sql")
        st.divider()
        st.markdown("#### ③ 安全与业务校验")
        st.write(
            "SQL Safety："
            + ("PASS" if details.get("sql_safety_valid") else "无可用状态")
        )
        st.write(
            "Business Validator："
            + ("PASS" if details.get("business_validation_valid") else "无可用状态")
        )
        st.write(f"Repair：{'Yes' if details.get('repair_triggered') else 'No'}")
        for warning in details.get("business_warnings", []):
            st.warning(f"Warning：{warning}")
        st.divider()
        st.markdown("#### ④ 数据库查询")
        dataframe = details.get("dataframe")
        row_count = len(dataframe) if isinstance(dataframe, pd.DataFrame) else "未保存"
        st.write("SQLite：Read-only")
        st.write(
            "执行状态："
            + ("Success" if details.get("status", "success") == "success" else "Failed")
        )
        st.write(f"返回行数：{row_count}")
        st.write(f"结果截断：{'Yes' if details.get('truncated') else 'No'}")
        if details.get("error_message"):
            st.error(
                f"{details.get('error_type') or 'TaskError'}: "
                f"{details['error_message']}"
            )
        st.divider()
        st.markdown("#### ⑤ 分析依据")
        show_evidence(details)


def render_assistant_message(message: dict[str, Any]) -> None:
    kind = message.get("kind", "result")
    if kind == "error":
        st.error(message["content"])
        return
    if kind == "clarification":
        st.markdown(message["content"])
        st.caption("需要补充信息后才能生成可靠查询。")
        return
    details = message.get("details")
    if not isinstance(details, dict):
        st.markdown(message["content"])
        return
    task_details = details.get("tasks")
    if not isinstance(task_details, list) or not task_details:
        task_details = [details]

    for index, task in enumerate(task_details, 1):
        task_question = task.get("question") or f"任务 {index}"
        st.subheader(f"分析任务 {index}：{task_question}")
        if task.get("status", "success") == "failed":
            st.error(
                f"任务执行失败：{task.get('error_type') or 'TaskError'}: "
                f"{task.get('error_message') or '未知错误'}"
            )
        else:
            show_query_data(task)
            chart_path = task.get("chart_path")
            if chart_path and Path(chart_path).is_file():
                st.image(chart_path, use_container_width=True)
            elif chart_path:
                st.caption("历史图表文件当前不可用，未重新生成。")
            if task.get("chart_error"):
                st.warning(task["chart_error"])

        process_label = (
            "问题理解与执行过程"
            if len(task_details) == 1
            else f"分析任务 {index} 的执行过程"
        )
        show_execution_process(task, label=process_label)
        if index < len(task_details):
            st.divider()

    st.markdown("### 综合结论")
    st.markdown(message["content"])


def render_chat_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


def show_header() -> None:
    st.title("数据分析 Agent")
    st.caption("自然语言 · Text-to-SQL · Read-only Analytics")


def show_empty_state() -> None:
    with st.container(key="empty_state"):
        st.markdown("### 用自然语言查询门店经营数据")
        st.caption("可以从下方示例问题开始。")


def show_example_buttons() -> str | None:
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, (label, question) in zip(columns, EXAMPLE_QUESTIONS.items()):
        with column:
            if st.button(
                label,
                key=f"official_example_{label}",
                help=question,
                use_container_width=True,
            ):
                return question
    return None


def main() -> None:
    st.set_page_config(
        page_title="Data Analysis Agent", page_icon="📊", layout="wide"
    )
    initialize_session_state()
    apply_page_style()
    show_sidebar()
    show_header()
    render_chat_history()
    if not st.session_state["messages"]:
        show_empty_state()
    with st.container(key="composer"):
        example_question = show_example_buttons()
        typed_question = st.chat_input(
            "向数据分析 Agent 提问……", key="main_chat_input"
        )
    submitted_question = example_question or typed_question
    if submitted_question and submitted_question.strip():
        submit_question(submitted_question)


if __name__ == "__main__":
    main()
