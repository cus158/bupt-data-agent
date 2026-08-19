"""Conversational Streamlit presentation layer for the Text-to-SQL agent."""

from __future__ import annotations

import json
import logging
import os
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
    create_chart,
    run_agent,
)
from bupt_data_agent.conversation import context_display_summary
from bupt_data_agent.evidence import build_query_evidence
from bupt_data_agent.paths import DB_PATH, ENV_FILE, EVALUATION_DIR


EVALUATION_REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"
EXAMPLE_QUESTIONS = {
    "门店销售额": "查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    "Top SKU": "查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。",
    "退损分析": "查询各门店 2025 年上半年的退损情况，找出退损率超过 5% 的门店，画出各门店退损率对比图，并分析退损率较高门店的主要退款原因。",
    "季度增长": "比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。",
    "毛利下钻": "找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。",
}


def apply_page_style() -> None:
    """Apply a small amount of static layout polish without styling user content."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1100px;
            padding-top: 1.4rem;
            padding-bottom: 5rem;
        }
        div[data-testid="stChatMessage"] {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
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


def load_benchmark_summary() -> tuple[str, str]:
    """Read the existing report only; never rerun evaluation from the UI."""
    try:
        payload = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8-sig"))
        core = payload.get("core_benchmark", {})
        metrics = payload.get("metrics", {})
        return (
            f"{core['passed']} / {core['total']} PASS",
            f"{metrics['passed']} / {metrics['total_cases']} PASS",
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "报告不可用", "报告不可用"


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_example_text", None)


def _entity_labels(context: dict[str, Any], entity_type: str) -> list[str]:
    id_key, name_key = (
        ("store_id", "store_name")
        if entity_type == "stores"
        else ("product_id", "product_name")
    )
    return [
        " ".join(
            value
            for value in (str(item.get(id_key, "")), str(item.get(name_key, "")))
            if value
        )
        for item in context.get("entities", {}).get(entity_type, [])[:3]
    ]


def show_context_summary(context: dict[str, Any] | None) -> None:
    if not context:
        st.caption("暂无上下文")
        return

    stores = _entity_labels(context, "stores")
    products = _entity_labels(context, "products")
    time_context = context.get("time_context", {})
    time_label = " ".join(
        str(value)
        for value in (time_context.get("year"), time_context.get("period"))
        if value
    )
    metrics = [str(item) for item in context.get("metrics", [])[:3]]

    if stores:
        st.caption("门店")
        st.write("、".join(stores))
    if products:
        st.caption("SKU")
        st.write("、".join(products))
    if time_label:
        st.caption("时间")
        st.write(time_label)
    if metrics:
        st.caption("指标")
        st.write("、".join(metrics))
    if not any((stores, products, time_label, metrics)):
        st.caption(context_display_summary(context) or "暂无可复用实体")


def start_new_conversation() -> None:
    """Clear display history and Agent memory without running the Agent."""
    st.session_state["messages"] = []
    for key in (
        "last_turn_context",
        "selected_example_text",
        "analysis_result",
        "analysis_error",
        "clarification",
        "last_question",
    ):
        st.session_state.pop(key, None)


def show_sidebar() -> None:
    core_benchmark, extended_evaluation = load_benchmark_summary()
    with st.sidebar:
        st.title("Data Analysis Agent")
        if st.button("＋ 新对话", type="primary", use_container_width=True):
            start_new_conversation()
            st.rerun()

        st.divider()
        st.subheader("示例问题")
        for label, question in EXAMPLE_QUESTIONS.items():
            if st.button(label, key=f"example_{label}", use_container_width=True):
                st.session_state["selected_example_text"] = question

        selected_example = st.session_state.get("selected_example_text")
        if selected_example:
            st.caption("已选择示例（复制到下方输入框后发送）")
            st.code(selected_example, language=None, wrap_lines=True)

        st.divider()
        st.subheader("当前上下文")
        show_context_summary(st.session_state.get("last_turn_context"))
        if st.button("清除上下文记忆", use_container_width=True):
            st.session_state.pop("last_turn_context", None)
            st.rerun()

        st.divider()
        st.subheader("System")
        st.write(f"SQLite　{'Ready' if DB_PATH.is_file() else 'Unavailable'}")
        st.write(f"LLM　{configured_model_name()}")
        st.write("SQL　Read-only")
        st.write("Validator　Enabled")

        st.divider()
        st.subheader("Benchmark")
        st.write(f"Core：{core_benchmark}")
        st.write(f"Extended：{extended_evaluation}")
        st.caption("基于当前项目评测集")


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


def analyze_question(question: str) -> dict[str, Any]:
    """Run one new Agent turn and return a cached assistant display message."""
    previous_context = st.session_state.get("last_turn_context")
    try:
        result = run_agent(question, conversation_context=previous_context)
        if result.plan.status == "needs_clarification":
            return {
                "role": "assistant",
                "kind": "clarification",
                "content": result.plan.clarification_question or "请补充查询条件。",
                "ambiguity_type": result.plan.ambiguity_type,
            }
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
                    "对话上下文：本次问题使用了上一轮结构化查询上下文"
                    + (f"（{summary}）" if summary else "。"),
                )
        except Exception:
            logging.exception("Query evidence generation failed")
            evidence_error = "查询依据生成失败，但不影响实际查询结果。"

        validation = result.business_validation
        details = {
            "question": question,
            "dataframe": result.query_result.dataframe.copy(),
            "truncated": result.query_result.truncated,
            "sql": result.plan.sql,
            "reasoning_summary": result.plan.reasoning_summary,
            "chart_type": result.plan.chart_type,
            "chart_path": chart_path,
            "chart_error": chart_error,
            "evidence": evidence,
            "evidence_error": evidence_error,
            "business_validation_valid": bool(validation and validation.valid),
            "business_warnings": (
                [warning.message for warning in validation.warnings]
                if validation
                else []
            ),
            "repair_triggered": result.repair_triggered,
            "first_error_type": result.first_error_type,
            "conversation_context_used": result.conversation_context_used,
            "model": configured_model_name(),
        }
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
        elif any(token in name for token in ("quantity", "count", "number")):
            formatters[column] = lambda value: (
                "—" if pd.isna(value) else f"{float(value):,.0f}"
            )
    return dataframe.style.format(formatters, na_rep="—")


def show_query_data(details: dict[str, Any]) -> None:
    dataframe = details["dataframe"]
    if dataframe.empty:
        st.info("查询执行成功，但没有符合条件的数据。")
        return
    st.dataframe(
        _styled_dataframe(dataframe),
        use_container_width=True,
        hide_index=True,
        height=min(520, 90 + 35 * min(len(dataframe), 12)),
    )
    if details["truncated"]:
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

    status_badges = {
        "SQL 安全检查已通过": "✓ SQL Safety",
        "Business Rule Validator 已通过": "✓ Business Rules",
        "SQLite 只读查询已执行": "✓ SQLite Read-only",
        "结果来自实际数据库查询": "✓ Database Result",
    }
    badges = [
        status_badges[item]
        for item in evidence.get("status", [])
        if item in status_badges
    ]
    if details.get("conversation_context_used"):
        badges.append("✓ Previous Context Used")
    st.caption("　".join(badges))


def show_execution_details(details: dict[str, Any]) -> None:
    st.write("SQL Safety：PASS")
    st.write(
        "Business Validator："
        + ("PASS" if details["business_validation_valid"] else "无可用状态")
    )
    st.write("SQLite：Read-only")
    st.write(f"Query Rows：{len(details['dataframe']):,}")
    st.write(f"Truncated：{'Yes' if details['truncated'] else 'No'}")
    st.write(f"Auto Repair：{'Yes' if details['repair_triggered'] else 'No'}")
    if details["repair_triggered"] and details.get("first_error_type"):
        st.write(f"First Error：{details['first_error_type']}")
    st.write(
        "Conversation Context："
        + ("Used" if details["conversation_context_used"] else "Not Used")
    )
    st.write(f"Chart Type：{details['chart_type']}")
    st.write(f"Model：{details['model']}")
    for warning in details.get("business_warnings", []):
        st.warning(f"Business Warning：{warning}")
    if details.get("chart_error"):
        st.warning(details["chart_error"])
    st.markdown("**分析说明**")
    st.write(details["reasoning_summary"])


def render_assistant_message(message: dict[str, Any]) -> None:
    kind = message.get("kind", "result")
    if kind == "error":
        st.error(message["content"])
        return

    st.markdown(message["content"])
    if kind == "clarification":
        st.caption("需要补充信息后才能生成可靠查询。")
        return

    details = message["details"]
    chart_path = details.get("chart_path")
    if chart_path and Path(chart_path).is_file():
        st.image(chart_path, use_container_width=True)

    with st.expander("查看查询数据"):
        show_query_data(details)
    with st.expander("查看分析依据"):
        show_evidence(details)
    with st.expander("查看生成 SQL"):
        st.code(details["sql"], language="sql")
    with st.expander("查看 Agent 执行详情"):
        show_execution_details(details)


def render_chat_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


def show_header() -> None:
    st.title("Data Analysis Agent")
    st.caption("Markdown-grounded · Text-to-SQL · Read-only Analytics")
    st.write("通过自然语言查询业务数据，并由 Agent 完成 SQL生成、安全校验、分析与可视化。")
    database_status = "Ready" if DB_PATH.is_file() else "Unavailable"
    st.caption(
        f"SQLite：{database_status}　·　LLM：{llm_configuration_status()}　·　"
        "SQL：Read-only　·　Business Validator：Enabled"
    )


def main() -> None:
    st.set_page_config(
        page_title="Data Analysis Agent",
        page_icon="📊",
        layout="wide",
    )
    initialize_session_state()
    apply_page_style()
    show_sidebar()
    show_header()
    render_chat_history()

    prompt = st.chat_input("向数据分析 Agent 提问……")
    if prompt and prompt.strip():
        question = prompt.strip()
        user_message = {"role": "user", "content": question}
        st.session_state["messages"].append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.status("Agent 正在分析数据……", expanded=True) as status:
                st.write(
                    "正在执行：业务理解 → SQL生成 → 安全与业务规则校验 → "
                    "数据库查询 → 结果分析"
                )
                assistant_message = analyze_question(question)
                if assistant_message["kind"] == "error":
                    status.update(label="分析未完成", state="error", expanded=False)
                elif assistant_message["kind"] == "clarification":
                    status.update(label="需要补充信息", state="complete", expanded=False)
                else:
                    status.update(label="分析完成", state="complete", expanded=False)
            render_assistant_message(assistant_message)
        st.session_state["messages"].append(assistant_message)


if __name__ == "__main__":
    main()
