"""Command-line interface for the minimal Text-to-SQL agent."""

from __future__ import annotations

from .agent import (
    AgentError,
    ConfigurationError,
    MAX_RESULT_ROWS,
    create_chart,
    run_agent,
)


def section(title: str) -> None:
    print(f"\n========== {title} ==========")


def main() -> int:
    try:
        question = input("请输入问题：\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return 1

    if not question:
        print("问题不能为空。")
        return 1

    section("问题")
    print(question)

    try:
        result = run_agent(question)
    except ConfigurationError as exc:
        section("配置错误")
        print(exc)
        print("请复制 .env.example 为 .env，并配置 LLM_API_KEY 和 LLM_MODEL。")
        return 2
    except AgentError as exc:
        section("运行错误")
        print(exc)
        return 3
    except Exception as exc:
        section("未预期错误")
        print(f"{type(exc).__name__}: {exc}")
        return 4

    if result.plan.status == "needs_clarification":
        section("需要补充信息")
        print(result.plan.clarification_question)
        return 0

    if result.query_result is None:
        section("运行错误")
        print("查询未返回可展示的结果。")
        return 3

    section("生成 SQL")
    print(result.plan.sql)

    section("SQL解释")
    print(result.plan.reasoning_summary)

    section("查询结果")
    dataframe = result.query_result.dataframe
    if dataframe.empty:
        print("（空结果）")
    else:
        print(
            dataframe.to_string(
                index=False,
                na_rep="NULL",
                max_rows=MAX_RESULT_ROWS,
            )
        )
    if result.query_result.truncated:
        print(f"\n结果超过 {MAX_RESULT_ROWS} 行，当前仅展示前 {MAX_RESULT_ROWS} 行。")

    section("分析结论")
    print(result.conclusion)

    section("图表")
    try:
        chart_path = create_chart(dataframe, result.plan.chart_type, question)
    except Exception as exc:
        print(f"图表生成失败：{type(exc).__name__}: {exc}")
    else:
        if chart_path:
            print(chart_path)
        else:
            print("当前结果不需要或不适合生成图表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
