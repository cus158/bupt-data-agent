"""Streamlit presentation layer for the existing Text-to-SQL agent."""

from __future__ import annotations

import json
import logging
import os
import re
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
    "门店成交销额": "查询2025年上半年各门店的成交销额，并按销额降序排列。",
    "华东 O2O Top SKU": "查询2025年上半年华东战区即时零售动销最好的3个SKU。",
    "高退损门店": "计算2025年上半年各门店退损率，找出超过5%的门店，并分析主要退款原因。",
    "Q1 / Q2 增长": "比较2025年Q1和Q2各门店成交销额，找出增长超过10%的门店。",
    "毛利率下钻": "找出Q2相比Q1成交销额增长但毛利率下降的门店，并分析是否可能存在低毛利SKU放量拖累。",
}


def apply_page_style() -> None:
    """Apply a small amount of static layout polish without styling user content."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 0.7rem;
            padding: 0.85rem 1rem;
            min-height: 7rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.75rem;
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
        "Configured"
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
        core_text = f"{core['passed']} / {core['total']} PASS"
        extended_text = f"{metrics['passed']} / {metrics['total_cases']} PASS"
        return core_text, extended_text
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "报告不可用", "报告不可用"


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
        st.caption("暂无对话上下文")
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


def show_sidebar() -> None:
    core_benchmark, extended_evaluation = load_benchmark_summary()
    with st.sidebar:
        st.header("系统信息")
        st.caption("Database")
        st.write("business.db")
        st.caption("Model")
        st.write(configured_model_name())
        st.caption("Mode")
        st.write("SQLite read-only")

        st.divider()
        st.subheader("Agent Pipeline")
        st.markdown(
            "① 业务语义理解  \n"
            "② Text-to-SQL  \n"
            "③ SQL Safety  \n"
            "④ Business Validation  \n"
            "⑤ SQLite Execution  \n"
            "⑥ Analysis & Visualization"
        )

        st.divider()
        st.subheader("Conversation Context")
        st.toggle("启用上下文追问", value=True, key="context_enabled")
        if st.button(
            "清除对话上下文",
            use_container_width=True,
            key="clear_conversation_context",
        ):
            st.session_state.pop("last_turn_context", None)
        show_context_summary(st.session_state.get("last_turn_context"))

        st.divider()
        st.subheader("当前项目评测集")
        st.caption("Core Benchmark")
        st.write(core_benchmark)
        st.caption("Extended Evaluation")
        st.write(extended_evaluation)


def show_system_status() -> None:
    cards = (
        ("🗄 SQLite", "Ready" if DB_PATH.is_file() else "Unavailable", "business.db"),
        ("🤖 LLM", llm_configuration_status(), configured_model_name()),
        ("🔒 SQL Safety", "Read Only", "Validated before execution"),
        ("✓ Business Rules", "Enabled", "Deterministic validator"),
    )
    for column, (label, value, detail) in zip(st.columns(4), cards):
        with column:
            st.metric(label=label, value=value, help=detail)
            st.caption(detail)


def apply_selected_example() -> None:
    selected = st.session_state.get("selected_example")
    if selected:
        st.session_state["question_input"] = EXAMPLE_QUESTIONS[selected]


def execute_analysis(question: str) -> None:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_error", None)
    st.session_state.pop("clarification", None)
    st.session_state["last_question"] = question

    try:
        previous_context = (
            st.session_state.get("last_turn_context")
            if st.session_state.get("context_enabled", True)
            else None
        )
        result = run_agent(question, conversation_context=previous_context)
        if result.plan.status == "needs_clarification":
            st.session_state["clarification"] = {
                "question": question,
                "clarification_question": result.plan.clarification_question,
                "ambiguity_type": result.plan.ambiguity_type,
            }
            return
        if result.query_result is None or result.plan.sql is None:
            raise AgentError("Ready plan did not return an executable query result")

        chart_path = None
        chart_error = None
        if result.plan.chart_type != "none":
            try:
                chart_path = create_chart(
                    result.query_result.dataframe,
                    result.plan.chart_type,
                    question,
                )
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
                evidence["notes"].insert(
                    0,
                    "对话上下文：本次问题使用了上一轮结构化查询上下文"
                    + (f"（{summary}）" if summary else "。"),
                )
        except Exception:
            logging.exception("Query evidence generation failed")
            evidence_error = "查询依据生成失败，但不影响实际查询结果。"

        st.session_state["analysis_result"] = {
            "question": question,
            "result": result,
            "chart_path": chart_path,
            "chart_error": chart_error,
            "evidence": evidence,
            "evidence_error": evidence_error,
            "model": configured_model_name(),
        }
        if st.session_state.get("context_enabled", True):
            st.session_state["last_turn_context"] = result.turn_context
    except ConfigurationError:
        st.session_state["analysis_error"] = "模型配置不可用，请检查 .env。"
    except SQLSafetyError:
        st.session_state["analysis_error"] = "生成的 SQL 未通过安全校验。"
    except BusinessRuleValidationError as exc:
        messages = "；".join(issue.message for issue in exc.result.violations)
        st.session_state["analysis_error"] = f"生成的查询未通过业务规则校验：{messages}"
    except SQLExecutionError:
        st.session_state["analysis_error"] = "查询执行失败，请检查问题后重试。"
    except LLMResponseError:
        st.session_state["analysis_error"] = "模型返回格式异常，请稍后重试。"
    except openai.AuthenticationError:
        st.session_state["analysis_error"] = "模型服务认证失败，请检查 .env 配置。"
    except openai.APITimeoutError:
        st.session_state["analysis_error"] = "模型服务请求超时，请稍后重试。"
    except openai.APIConnectionError:
        st.session_state["analysis_error"] = "模型服务请求失败，请检查网络后重试。"
    except openai.APIStatusError as exc:
        st.session_state["analysis_error"] = (
            f"模型服务请求失败（HTTP {exc.status_code}），请稍后重试。"
        )
    except AgentError:
        st.session_state["analysis_error"] = "分析执行失败，请稍后重试。"
    except Exception:
        logging.exception("Unexpected Streamlit analysis error")
        st.session_state["analysis_error"] = "分析过程中发生未预期错误。"


def _first_matching_column(dataframe: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    columns = {str(column).lower(): str(column) for column in dataframe.columns}
    return next((columns[name] for name in names if name in columns), None)


def _numeric_values(dataframe: pd.DataFrame, column: str | None) -> pd.Series:
    if not column:
        return pd.Series(dtype="float64")
    return pd.to_numeric(dataframe[column], errors="coerce").dropna()


def _format_amount(value: float) -> str:
    return f"{value:,.2f}"


def _format_rate(value: float, column: str) -> str:
    percentage = value if "pct" in column.lower() or abs(value) > 1 else value * 100
    return f"{percentage:.2f}%"


def build_kpi_cards(dataframe: pd.DataFrame, question: str) -> list[tuple[str, str]]:
    """Build conservative display KPIs from actual result columns only."""
    if dataframe.empty:
        return []

    cards: list[tuple[str, str]] = []
    store_column = _first_matching_column(dataframe, ("store_id",))
    product_column = _first_matching_column(dataframe, ("product_id",))
    ranking_question = bool(re.search(r"最高|最好|top|降序|排名", question, re.IGNORECASE))
    if ranking_question and product_column:
        cards.append(("Top SKU", str(dataframe.iloc[0][product_column])))
    elif ranking_question and store_column:
        cards.append(("Top 门店", str(dataframe.iloc[0][store_column])))

    amount_column = _first_matching_column(
        dataframe,
        (
            "sales_amount",
            "total_sales_amount",
            "q2_sales_amount",
            "refund_amount_sum",
            "refund_amount",
            "gross_profit",
        ),
    )
    amount_values = _numeric_values(dataframe, amount_column)
    if amount_column and not amount_values.empty:
        amount_name = amount_column.lower()
        if "refund" in amount_name:
            label = "最高退款金额"
        elif "profit" in amount_name:
            label = "最高毛利额"
        else:
            label = "最高成交销额"
        cards.append((label, _format_amount(float(amount_values.max()))))

    rate_column = _first_matching_column(
        dataframe,
        (
            "refund_rate",
            "refund_loss_rate",
            "loss_rate",
            "gross_margin_rate",
            "margin_rate",
            "growth_rate",
            "growth_pct",
            "sales_growth",
        ),
    )
    rate_values = _numeric_values(dataframe, rate_column)
    if rate_column and not rate_values.empty:
        rate_name = rate_column.lower()
        if "refund" in rate_name or "loss" in rate_name:
            label = "最高退损率"
        elif "margin" in rate_name:
            label = "最高毛利率"
        elif "growth" in rate_name:
            label = "最高增长率"
        else:
            label = "最高比例"
        cards.append((label, _format_rate(float(rate_values.max()), rate_column)))

    if store_column:
        count = int(dataframe[store_column].dropna().nunique())
        cards.append(("返回门店数", f"{count:,}"))
    elif product_column:
        count = int(dataframe[product_column].dropna().nunique())
        cards.append(("返回 SKU 数", f"{count:,}"))
    else:
        cards.append(("返回记录数", f"{len(dataframe):,}"))

    deduplicated: list[tuple[str, str]] = []
    for item in cards:
        if item[0] not in {label for label, _ in deduplicated}:
            deduplicated.append(item)
    return deduplicated[:4]


def show_kpis(dataframe: pd.DataFrame, question: str) -> None:
    cards = build_kpi_cards(dataframe, question)
    if not cards:
        return
    st.subheader("关键指标")
    for column, (label, value) in zip(st.columns(len(cards)), cards):
        with column:
            st.metric(label, value)


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


def show_query_output(state: dict[str, Any]) -> None:
    result = state["result"]
    dataframe = result.query_result.dataframe
    chart_path = state.get("chart_path")

    if dataframe.empty:
        st.info("查询执行成功，但没有符合条件的数据。")
        return

    def show_table() -> None:
        st.dataframe(
            _styled_dataframe(dataframe),
            use_container_width=True,
            hide_index=True,
            height=min(520, 90 + 35 * min(len(dataframe), 12)),
        )
        if result.query_result.truncated:
            st.warning(f"结果较多，仅展示前 {MAX_RESULT_ROWS} 行。")

    if chart_path:
        chart_column, table_column = st.columns([1.05, 1])
        with chart_column:
            st.subheader("可视化")
            st.image(str(chart_path), use_container_width=True)
        with table_column:
            st.subheader("查询结果")
            show_table()
    else:
        st.subheader("查询结果")
        show_table()

    if state.get("chart_error"):
        st.warning(state["chart_error"])


def _show_evidence_card(title: str, values: list[str], *, code: bool = False) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for value in values:
            rendered = f"`{value}`" if code else value
            st.markdown(f"- {rendered}")


def show_evidence(state: dict[str, Any]) -> None:
    evidence = state.get("evidence")
    if not evidence:
        if state.get("evidence_error"):
            st.warning(state["evidence_error"])
        return

    st.subheader("查询依据")
    sections: list[tuple[str, list[str], bool]] = []
    if evidence["tables"]:
        sections.append(("🗄 使用数据", evidence["tables"], True))
    if evidence["business_terms"]:
        sections.append(("📊 业务口径", evidence["business_terms"], False))
    if evidence["time_rules"]:
        sections.append(("🕒 时间范围", evidence["time_rules"], False))
    operations = evidence["filters"] + evidence["aggregation"]
    if operations:
        sections.append(("🔍 筛选与聚合", operations, False))

    if sections:
        for column, (title, values, code) in zip(st.columns(len(sections)), sections):
            with column:
                _show_evidence_card(title, values, code=code)

    for note in evidence["notes"]:
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
        for item in evidence["status"]
        if item in status_badges
    ]
    if state["result"].conversation_context_used:
        badges.append("✓ Previous Context Used")
    st.caption("　".join(badges))
    st.caption("业务口径展示来自 knowledge Markdown；完整 SQL 是最终技术证据。")


def show_technical_details(state: dict[str, Any]) -> None:
    result = state["result"]
    sql_tab, reasoning_tab, execution_tab = st.tabs(
        ["生成 SQL", "分析说明", "执行详情"]
    )
    with sql_tab:
        st.code(result.plan.sql, language="sql")
    with reasoning_tab:
        st.write(result.plan.reasoning_summary)
    with execution_tab:
        validation = result.business_validation
        st.write("SQL Safety：通过")
        st.write(
            "Business Validator："
            + ("通过" if validation and validation.valid else "无可用状态")
        )
        if validation and validation.warnings:
            for warning in validation.warnings:
                st.warning(warning.message)
        st.write(f"SQL 自动修复：{'是' if result.repair_triggered else '否'}")
        if result.repair_triggered and result.first_error_type:
            st.write(f"首次失败类型：{result.first_error_type}")
        st.write(
            f"对话上下文：{'已使用' if result.conversation_context_used else '未使用'}"
        )
        st.write(f"结果行数：{len(result.query_result.dataframe):,}")
        st.write(f"结果截断：{'是' if result.query_result.truncated else '否'}")
        st.write(f"图表类型：{result.plan.chart_type}")
        st.write(f"当前模型：{state['model']}")


def show_result() -> None:
    error = st.session_state.get("analysis_error")
    if error:
        st.error(error)
        return

    clarification = st.session_state.get("clarification")
    if clarification:
        st.divider()
        with st.container(border=True):
            st.subheader("需要补充信息")
            st.info(clarification["clarification_question"])
            if clarification.get("ambiguity_type"):
                st.caption(f"歧义类型：{clarification['ambiguity_type']}")
        return

    state = st.session_state.get("analysis_result")
    if not state:
        return

    result = state["result"]
    dataframe = result.query_result.dataframe

    st.divider()
    with st.container(border=True):
        st.subheader("分析结论")
        st.markdown(result.conclusion)

    show_kpis(dataframe, state["question"])
    show_query_output(state)

    st.divider()
    show_evidence(state)

    st.divider()
    st.subheader("技术详情")
    show_technical_details(state)


def show_footer() -> None:
    st.divider()
    st.caption("Built with Python · SQLite · pandas · matplotlib · DeepSeek")
    st.caption("SQL execution is read-only and validated before execution.")


def main() -> None:
    st.set_page_config(
        page_title="智能数据分析 Agent",
        page_icon="📊",
        layout="wide",
    )
    apply_page_style()
    show_sidebar()

    st.title("智能数据分析 Agent")
    st.caption("Natural Language → Business Understanding → Text-to-SQL → Trusted Analytics")
    st.write("通过自然语言完成业务查询、SQL 生成、安全校验、数据分析与可视化。")

    show_system_status()

    st.subheader("试试这些问题")
    st.selectbox(
        "选择示例",
        list(EXAMPLE_QUESTIONS),
        index=None,
        placeholder="选择一个示例以填充输入框",
        key="selected_example",
        on_change=apply_selected_example,
        label_visibility="collapsed",
    )

    with st.container(border=True):
        st.subheader("请输入业务问题")
        question = st.text_area(
            "业务问题",
            key="question_input",
            height=120,
            placeholder="例如：查询2025年上半年华东战区即时零售成交销额最高的3个SKU",
            label_visibility="collapsed",
        )
        analyze_clicked = st.button(
            "开始分析",
            type="primary",
            use_container_width=True,
            key="run_analysis",
        )

    if analyze_clicked:
        cleaned_question = question.strip()
        if not cleaned_question:
            st.session_state.pop("analysis_result", None)
            st.session_state["analysis_error"] = "请输入需要分析的问题。"
        else:
            with st.status("Agent 正在分析...", expanded=True) as status:
                st.write(
                    "正在执行：业务理解 → SQL 生成 → 安全与业务规则校验 → "
                    "数据库查询 → 结果分析"
                )
                execute_analysis(cleaned_question)
                if st.session_state.get("analysis_error"):
                    status.update(label="分析未完成", state="error", expanded=False)
                elif st.session_state.get("clarification"):
                    status.update(label="需要补充信息", state="complete", expanded=False)
                else:
                    status.update(label="分析完成", state="complete", expanded=False)

    show_result()
    show_footer()


if __name__ == "__main__":
    main()
