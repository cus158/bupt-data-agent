"""Command-line interface for the task-based Text-to-SQL agent."""

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

    if result.task_results:
        for index, task_result in enumerate(result.task_results, 1):
            section(f"分析任务 {index}：{task_result.task.question}")
            print("SQL：")
            print(task_result.task.sql)
            print("\nSQL解释：")
            print(task_result.task.reasoning_summary)
            if task_result.status == "failed":
                print(
                    f"\n执行失败：{task_result.error_type}: "
                    f"{task_result.error_message}"
                )
                continue

            query_result = task_result.query_result
            assert query_result is not None
            print("\n查询结果：")
            if query_result.dataframe.empty:
                print("（空结果）")
            else:
                print(
                    query_result.dataframe.to_string(
                        index=False,
                        na_rep="NULL",
                        max_rows=MAX_RESULT_ROWS,
                    )
                )
            if query_result.truncated:
                print(
                    f"\n结果超过 {MAX_RESULT_ROWS} 行，当前仅展示前 "
                    f"{MAX_RESULT_ROWS} 行。"
                )
            print("\n图表：")
            if task_result.chart_path:
                print(task_result.chart_path)
            elif task_result.chart_error:
                print(f"图表生成失败：{task_result.chart_error}")
            else:
                print("当前任务不需要或不适合生成图表。")
    else:
        # Compatibility for older AgentResult producers.
        if result.query_result is None:
            section("运行错误")
            print("查询未返回可展示的结果。")
            return 3
        section("分析任务 1")
        print(result.plan.sql)
        dataframe = result.query_result.dataframe
        print(dataframe.to_string(index=False, na_rep="NULL", max_rows=MAX_RESULT_ROWS))
        chart_path = create_chart(dataframe, result.plan.chart_type, question)
        if chart_path:
            print(chart_path)

    section("综合结论")
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
