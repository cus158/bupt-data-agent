"""Streamlit UI for the existing Text-to-SQL agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv

from agent import (
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
from evidence import build_query_evidence


PROJECT_DIR = Path(__file__).resolve().parent
EXAMPLE_QUESTIONS = [
    "查询2025年上半年各门店的成交销额，并按销额降序排列。",
    "查询2025年上半年华东战区即时零售动销最好的3个SKU。",
    "计算2025年上半年各门店退损率，找出超过5%的门店，并分析主要退款原因。",
    "比较2025年Q1和Q2各门店成交销额，找出增长超过10%的门店。",
    "找出Q2相比Q1成交销额增长但毛利率下降的门店，并分析是否可能存在低毛利SKU放量拖累。",
]


def configured_model_name() -> str:
    load_dotenv(PROJECT_DIR / ".env", override=False)
    return os.getenv("LLM_MODEL", "").strip() or "未配置"


def show_sidebar() -> None:
    with st.sidebar:
        st.header("系统状态")
        st.write("数据库：SQLite")
        st.write(f"LLM：{configured_model_name()}")
        st.write("知识来源：business_terms.md + 数据表说明")
        st.write("SQL模式：只读安全执行")
        st.divider()
        st.subheader("分析流程")
        st.markdown(
            "自然语言  \n"
            "↓  \n"
            "业务知识 + Schema  \n"
            "↓  \n"
            "LLM Text-to-SQL  \n"
            "↓  \n"
            "SQL 安全校验  \n"
            "↓  \n"
            "SQLite  \n"
            "↓  \n"
            "分析与可视化"
        )


def apply_selected_example() -> None:
    selected = st.session_state.get("selected_example")
    if selected:
        st.session_state["question_input"] = selected


def execute_analysis(question: str) -> None:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_error", None)
    st.session_state.pop("clarification", None)
    st.session_state["last_question"] = question

    try:
        result = run_agent(question)
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
    except ConfigurationError:
        st.session_state["analysis_error"] = (
            "未检测到 LLM API 配置，请检查 .env。"
        )
    except SQLSafetyError:
        st.session_state["analysis_error"] = "生成的 SQL 未通过安全校验。"
    except BusinessRuleValidationError as exc:
        messages = "；".join(issue.message for issue in exc.result.violations)
        st.session_state["analysis_error"] = f"生成的 SQL 未通过业务规则校验：{messages}"
    except SQLExecutionError as exc:
        st.session_state["analysis_error"] = f"SQL 执行失败：{exc}"
    except LLMResponseError:
        st.session_state["analysis_error"] = (
            "模型返回格式异常，请稍后重试或检查模型兼容性。"
        )
    except openai.AuthenticationError:
        st.session_state["analysis_error"] = (
            "LLM API 认证失败，请检查 .env 中的配置。"
        )
    except openai.APITimeoutError:
        st.session_state["analysis_error"] = "LLM API 请求超时，请稍后重试。"
    except openai.APIConnectionError:
        st.session_state["analysis_error"] = (
            "无法连接 LLM API，请检查网络后重试。"
        )
    except openai.APIStatusError as exc:
        st.session_state["analysis_error"] = (
            f"LLM API 请求失败（HTTP {exc.status_code}），请稍后重试。"
        )
    except AgentError:
        st.session_state["analysis_error"] = "分析执行失败，请稍后重试。"
    except Exception:
        logging.exception("Unexpected Streamlit analysis error")
        st.session_state["analysis_error"] = (
            "分析过程中发生未预期错误，请查看终端日志。"
        )


def show_result() -> None:
    error = st.session_state.get("analysis_error")
    if error:
        st.error(error)
        return

    clarification = st.session_state.get("clarification")
    if clarification:
        st.divider()
        st.subheader("需要补充信息")
        st.info(clarification["clarification_question"])
        return

    state = st.session_state.get("analysis_result")
    if not state:
        return

    result = state["result"]
    dataframe = result.query_result.dataframe

    st.divider()
    st.subheader("分析结论")
    st.markdown(result.conclusion)

    evidence = state.get("evidence")
    if evidence:
        st.subheader("查询依据")
        with st.expander("查看查询依据", expanded=True):
            left, right = st.columns(2)
            with left:
                st.markdown("**使用的数据表**")
                for table in evidence["tables"]:
                    st.markdown(f"- `{table}`")

                if evidence["business_terms"]:
                    st.markdown("**业务指标与术语**")
                    for term in evidence["business_terms"]:
                        st.markdown(f"- {term}")

            with right:
                if evidence["time_rules"]:
                    st.markdown("**时间口径**")
                    for rule in evidence["time_rules"]:
                        st.markdown(f"- {rule}")

                operations = evidence["filters"] + evidence["aggregation"]
                if operations:
                    st.markdown("**筛选与聚合**")
                    for operation in operations:
                        st.markdown(f"- {operation}")

            for note in evidence["notes"]:
                st.info(note)
            for warning in evidence.get("business_warnings", []):
                st.warning(f"业务规则提醒：{warning}")
            st.caption(" · ".join(f"✓ {item}" for item in evidence["status"]))
            st.caption(
                "业务口径展示来自 knowledge Markdown；完整 SQL 仍是最终技术证据。"
            )
    if state.get("evidence_error"):
        st.warning(state["evidence_error"])

    st.subheader("查询结果")
    if dataframe.empty:
        st.info("查询执行成功，但没有符合条件的数据。")
    else:
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
        if result.query_result.truncated:
            st.warning(f"结果较多，仅展示前 {MAX_RESULT_ROWS} 行。")

    chart_path = state.get("chart_path")
    if chart_path:
        st.subheader("图表")
        st.image(str(chart_path), use_container_width=True)
    if state.get("chart_error"):
        st.warning(state["chart_error"])

    with st.expander("查看生成的 SQL"):
        st.code(result.plan.sql, language="sql")

    with st.expander("查看 Agent 推理摘要"):
        st.write(result.plan.reasoning_summary)

    with st.expander("执行详情"):
        st.write(f"SQL 自动修复：{'是' if result.repair_triggered else '否'}")
        if (
            result.repair_triggered
            and result.first_error_type == "BusinessRuleValidationError"
        ):
            st.write("首次 SQL 未通过业务规则校验 → 已自动修复一次")
        validation = result.business_validation
        if validation:
            st.write(f"业务规则校验：{'通过' if validation.valid else '未通过'}")
            st.write(f"业务规则提醒：{len(validation.warnings)} 条")
        st.write(f"结果截断：{'是' if result.query_result.truncated else '否'}")
        st.write(f"图表类型：{result.plan.chart_type}")
        st.write(f"当前模型：{state['model']}")


def main() -> None:
    st.set_page_config(
        page_title="智能数据分析 Agent",
        page_icon="📊",
        layout="wide",
    )
    show_sidebar()

    st.title("智能数据分析 Agent")
    st.caption("基于业务知识与数据库 Schema 的自然语言 Text-to-SQL 分析")
    st.markdown(
        "自然语言提问 → 自动理解业务术语 → 生成只读 SQL → "
        "查询 SQLite → 返回分析结论与可视化"
    )

    st.selectbox(
        "示例问题",
        EXAMPLE_QUESTIONS,
        index=None,
        placeholder="选择一个示例以填充输入框",
        key="selected_example",
        on_change=apply_selected_example,
    )
    question = st.text_area(
        "请输入你的业务问题",
        key="question_input",
        height=120,
        placeholder="例如：查询2025年上半年各门店的成交销额，并按销额降序排列。",
    )

    if st.button("开始分析", type="primary", use_container_width=True):
        cleaned_question = question.strip()
        if not cleaned_question:
            st.session_state.pop("analysis_result", None)
            st.session_state["analysis_error"] = "请输入需要分析的问题。"
        else:
            with st.spinner("正在理解问题、生成安全 SQL 并执行分析..."):
                execute_analysis(cleaned_question)

    show_result()


if __name__ == "__main__":
    main()
