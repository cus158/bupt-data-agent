"""Real-LLM Text-to-SQL evaluation against deterministic Golden results."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Callable
from urllib.parse import urlparse

import openai
import pandas as pd
from dotenv import load_dotenv

from bupt_data_agent.agent import (
    AgentError,
    ConfigurationError,
    LLMResponseError,
    QueryResult,
    SQLExecutionError,
    SQLPlan,
    SQLSafetyError,
    execute_query,
    load_config,
    run_agent,
)
from bupt_data_agent.paths import ENV_FILE, EVALUATION_DIR, OUTPUTS_DIR


GOLDEN_PATH = EVALUATION_DIR / "golden_results.json"
REPORT_PATH = OUTPUTS_DIR / "online_test_report.txt"

QUESTIONS = [
    "查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。",
    "查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。",
    "查询各门店 2025 年上半年的退损情况，找出退损率超过 5% 的门店，画出各门店退损率对比图，并分析退损率较高门店的主要退款原因。",
    "比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。",
    "找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。",
]

CHART_EXPECTATIONS = ("bar", "none", "required", "required", "required")


def save_report(lines: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_base_url_description(base_url: str | None) -> str:
    if not base_url:
        return "SDK default"
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname:
        return "configured but invalid URL shape"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def sanitize_error(exc: Exception, api_key: str = "") -> str:
    message = " ".join(str(exc).split())
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", message)
    return message[:800]


def classify_error(exc: Exception, custom_base_url: bool) -> tuple[str, bool]:
    """Return (category, stop_remaining_questions)."""
    message = str(exc).lower()
    if isinstance(exc, ConfigurationError):
        return "配置缺失", True
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "API Key 问题", True
    if isinstance(exc, openai.RateLimitError):
        return "API Key 问题（额度或限流）", True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return ("base_url 问题" if custom_base_url else "网络问题"), True
    if isinstance(exc, openai.NotFoundError):
        return ("model name 问题" if "model" in message else "base_url 问题"), True
    if isinstance(exc, openai.BadRequestError):
        if "response_format" in message or "json" in message:
            return "response_format 不兼容", True
        if "model" in message:
            return "model name 问题", True
        return "API兼容问题", True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 401:
            return "API Key 问题", True
        if exc.status_code == 404:
            return ("model name 问题" if "model" in message else "base_url 问题"), True
        return "API兼容问题", True
    if isinstance(exc, LLMResponseError):
        return "LLM输出格式错误", False
    if isinstance(exc, SQLSafetyError):
        return "SQL生成问题（安全检查未通过）", False
    if isinstance(exc, SQLExecutionError):
        return "SQL生成问题（SQLite执行失败）", False
    if isinstance(exc, ValueError) and ("url" in message or "base" in message):
        return "base_url 问题", True
    if isinstance(exc, AgentError):
        return "SQL生成问题", False
    return "API兼容问题或未分类错误", True


def load_golden() -> dict:
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(
            f"Golden results not found: {GOLDEN_PATH}. "
            "Run 'python tests/smoke_test.py' first."
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def dimension_maps() -> tuple[dict[str, str], dict[str, str]]:
    stores = execute_query("SELECT store_id, store_name FROM store_info").dataframe
    products = execute_query("SELECT product_id, product_name FROM product_info").dataframe
    store_names = dict(zip(stores["store_name"], stores["store_id"]))
    product_names = dict(zip(products["product_name"], products["product_id"]))
    return store_names, product_names


def entity_ids(
    dataframe: pd.DataFrame,
    id_column: str,
    name_column: str,
    name_to_id: dict[str, str],
) -> pd.Series | None:
    if id_column in dataframe.columns:
        return dataframe[id_column].astype(str)
    if name_column in dataframe.columns:
        return dataframe[name_column].astype(str).map(name_to_id)

    known_ids = set(name_to_id.values())
    known_names = set(name_to_id)
    for column in dataframe.columns:
        values = dataframe[column].dropna().astype(str)
        if not values.empty and set(values).issubset(known_ids):
            return dataframe[column].astype(str)
        if not values.empty and set(values).issubset(known_names):
            return dataframe[column].astype(str).map(name_to_id)
    return None


def unique_sequence(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(value for value in values.dropna().astype(str)))


def numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(dataframe[column], errors="coerce")


RATE_COLUMN_TOKENS = (
    "rate",
    "ratio",
    "margin",
    "growth",
    "share",
    "pct",
    "percent",
    "percentage",
)


def normalize_rate(actual: float, expected: float) -> float:
    """Normalize ratio and percentage representations to the Golden scale."""
    if abs(expected) <= 1 and abs(actual) > 1:
        return actual / 100
    if abs(expected) > 1 and abs(actual) <= 1:
        return actual * 100
    return actual


def ordered_metric_columns(
    dataframe: pd.DataFrame,
    preferred_columns: tuple[str, ...],
    rate_metric: bool,
) -> list[str]:
    preferred = {name.lower() for name in preferred_columns}
    exact = [column for column in dataframe.columns if column.lower() in preferred]
    semantic = [
        column
        for column in dataframe.columns
        if column not in exact
        and rate_metric
        and any(token in column.lower() for token in RATE_COLUMN_TOKENS)
    ]
    remaining = [
        column for column in dataframe.columns if column not in exact and column not in semantic
    ]
    return exact + semantic + remaining


def find_metric_column(
    dataframe: pd.DataFrame,
    entities: pd.Series,
    expected: dict[str, float],
    tolerance: float,
    allow_percent_scale: bool = False,
    preferred_columns: tuple[str, ...] = (),
    relative_tolerance: float = 0.0,
) -> tuple[str, float] | None:
    for column in ordered_metric_columns(
        dataframe,
        preferred_columns,
        rate_metric=allow_percent_scale,
    ):
        values = numeric_series(dataframe, column)
        if values.notna().sum() == 0:
            continue
        matched = True
        used_percent_representation = False
        for entity, expected_value in expected.items():
            candidates = values[entities == entity].dropna().tolist()
            normalized = [
                normalize_rate(value, expected_value)
                if allow_percent_scale
                else value
                for value in candidates
            ]
            if not any(
                math.isclose(
                    value,
                    expected_value,
                    rel_tol=relative_tolerance,
                    abs_tol=tolerance,
                )
                for value in normalized
            ):
                matched = False
                break
            used_percent_representation = used_percent_representation or any(
                normalized_value != raw_value
                for raw_value, normalized_value in zip(candidates, normalized)
            )
        if matched:
            return column, 100.0 if used_percent_representation else 1.0
    return None


def rate_metric_diagnostics(
    dataframe: pd.DataFrame,
    entities: pd.Series,
    expected: dict[str, float],
    matched_column: tuple[str, float] | None,
    preferred_columns: tuple[str, ...],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> list[str]:
    if matched_column:
        column = matched_column[0]
    else:
        semantic_columns = ordered_metric_columns(
            dataframe,
            preferred_columns,
            rate_metric=True,
        )
        column = next(
            (
                candidate
                for candidate in semantic_columns
                if candidate.lower() in {name.lower() for name in preferred_columns}
                or any(token in candidate.lower() for token in RATE_COLUMN_TOKENS)
            ),
            None,
        )
    if not column:
        return ["比例指标诊断: 未找到语义候选列"]

    values = numeric_series(dataframe, column)
    details = [f"比例指标诊断列={column}"]
    for entity, expected_value in expected.items():
        actual_values = values[entities == entity].dropna().tolist()
        normalized_values = [
            normalize_rate(value, expected_value) for value in actual_values
        ]
        comparison = any(
            math.isclose(
                value,
                expected_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
            for value in normalized_values
        )
        details.append(
            f"{entity}: actual={actual_values}, normalized={normalized_values}, "
            f"expected={expected_value}, rel_tol={relative_tolerance}, "
            f"abs_tol={absolute_tolerance}, matched={comparison}"
        )
    return details


def find_reason_column(dataframe: pd.DataFrame, expected_reasons: set[str]) -> str | None:
    if "refund_reason" in dataframe.columns:
        return "refund_reason"
    for column in dataframe.columns:
        values = set(dataframe[column].dropna().astype(str))
        if expected_reasons.issubset(values):
            return column
    return None


def find_reason_amount_column(
    dataframe: pd.DataFrame,
    stores: pd.Series,
    reason_column: str,
    expected: dict[tuple[str, str], float],
) -> str | None:
    reasons = dataframe[reason_column].astype(str)
    for column in dataframe.columns:
        values = numeric_series(dataframe, column)
        if values.notna().sum() == 0:
            continue
        matched = True
        for (store_id, reason), expected_amount in expected.items():
            candidates = values[(stores == store_id) & (reasons == reason)].dropna().tolist()
            if not any(
                math.isclose(value, expected_amount, abs_tol=0.01) for value in candidates
            ):
                matched = False
                break
        if matched:
            return column
    return None


def compare_q1(
    dataframe: pd.DataFrame,
    golden: dict,
    store_names: dict[str, str],
    _product_names: dict[str, str],
    _conclusion: str,
) -> tuple[bool, list[str]]:
    expected_rows = golden["queries"]["example_1_store_sales"]["rows"]
    stores = entity_ids(dataframe, "store_id", "store_name", store_names)
    if stores is None:
        return False, ["未找到门店标识列"]
    expected_order = [row["store_id"] for row in expected_rows]
    actual_order = unique_sequence(stores)
    expected_amounts = {row["store_id"]: row["sales_amount"] for row in expected_rows}
    amount_column = find_metric_column(dataframe, stores, expected_amounts, 0.01)
    messages = [f"门店排名 actual={actual_order}, expected={expected_order}"]
    messages.append(f"销额列匹配={amount_column[0] if amount_column else 'none'}")
    return actual_order == expected_order and amount_column is not None, messages


def compare_q2(
    dataframe: pd.DataFrame,
    golden: dict,
    _store_names: dict[str, str],
    product_names: dict[str, str],
    _conclusion: str,
) -> tuple[bool, list[str]]:
    expected_rows = golden["queries"]["example_2_top_skus"]["rows"]
    products = entity_ids(dataframe, "product_id", "product_name", product_names)
    if products is None:
        return False, ["未找到 SKU 标识列"]
    expected_order = [row["product_id"] for row in expected_rows]
    actual_order = unique_sequence(products)
    expected_amounts = {row["product_id"]: row["sales_amount"] for row in expected_rows}
    amount_column = find_metric_column(dataframe, products, expected_amounts, 0.01)
    expected_quantities = {
        row["product_id"]: row["sales_quantity"] for row in expected_rows
    }
    quantity_column = find_metric_column(
        dataframe,
        products,
        expected_quantities,
        0.0,
        preferred_columns=("sales_quantity", "quantity", "total_quantity", "qty"),
    )
    expected_categories = {
        row["product_id"]: str(row["category"]) for row in expected_rows
    }
    category_column = None
    for column in dataframe.columns:
        if all(
            expected_category
            in dataframe.loc[products == product_id, column].dropna().astype(str).tolist()
            for product_id, expected_category in expected_categories.items()
        ):
            category_column = column
            break
    messages = [f"SKU排名 actual={actual_order}, expected={expected_order}"]
    messages.append(f"销额列匹配={amount_column[0] if amount_column else 'none'}")
    messages.append(f"销量列匹配={quantity_column[0] if quantity_column else 'none'}")
    messages.append(f"品类列匹配={category_column or 'none'}")
    return (
        actual_order == expected_order
        and amount_column is not None
        and quantity_column is not None
        and category_column is not None
    ), messages


def compare_q3(
    dataframe: pd.DataFrame,
    golden: dict,
    store_names: dict[str, str],
    _product_names: dict[str, str],
    conclusion: str,
) -> tuple[bool, list[str]]:
    rate_rows = [
        row
        for row in golden["queries"]["example_3_refund_rates"]["rows"]
        if row["refund_rate"] > 0.05
    ]
    reason_rows = golden["queries"]["example_3_refund_reasons"]["rows"]
    stores = entity_ids(dataframe, "store_id", "store_name", store_names)
    if stores is None:
        return False, ["未找到门店标识列"]
    expected_stores = [row["store_id"] for row in rate_rows]
    actual_stores = unique_sequence(stores)
    expected_rates = {row["store_id"]: row["refund_rate"] for row in rate_rows}
    rate_columns = (
        "refund_rate",
        "refund_ratio",
        "refund_loss_rate",
        "loss_rate",
    )
    rate_column = find_metric_column(
        dataframe,
        stores,
        expected_rates,
        tolerance=0.0001,
        allow_percent_scale=True,
        preferred_columns=rate_columns,
        relative_tolerance=0.001,
    )
    reasons = {row["refund_reason"] for row in reason_rows}
    reason_column = find_reason_column(dataframe, reasons)
    amount_column = None
    if reason_column:
        expected_amounts = {
            (row["store_id"], row["refund_reason"]): row["refund_amount"]
            for row in reason_rows
        }
        amount_column = find_reason_amount_column(
            dataframe,
            stores,
            reason_column,
            expected_amounts,
        )
    conclusion_ok = "质量问题" in conclusion and "不符合预期" in conclusion
    actual_high_stores: list[str] = []
    if rate_column:
        rate_values = numeric_series(dataframe, rate_column[0]) / rate_column[1]
        actual_high_stores = unique_sequence(stores[rate_values > 0.05])
    messages = [
        f"返回门店={actual_stores}",
        f"高退损门店 actual={actual_high_stores}, expected={expected_stores}",
        f"退损率列匹配={rate_column[0] if rate_column else 'none'}",
        f"退款原因列匹配={reason_column or 'none'}",
        f"退款金额列匹配={amount_column or 'none'}",
        f"结论包含主要原因={conclusion_ok}",
    ]
    messages.extend(
        rate_metric_diagnostics(
            dataframe,
            stores,
            expected_rates,
            rate_column,
            rate_columns,
            relative_tolerance=0.001,
            absolute_tolerance=0.0001,
        )
    )
    passed = (
        set(actual_high_stores) == set(expected_stores)
        and rate_column is not None
        and reason_column is not None
        and amount_column is not None
        and conclusion_ok
    )
    return passed, messages


def compare_q4(
    dataframe: pd.DataFrame,
    golden: dict,
    store_names: dict[str, str],
    _product_names: dict[str, str],
    _conclusion: str,
) -> tuple[bool, list[str]]:
    expected_rows = [
        row
        for row in golden["queries"]["example_4_quarterly_growth"]["rows"]
        if row["growth_rate"] > 0.10
    ]
    stores = entity_ids(dataframe, "store_id", "store_name", store_names)
    if stores is None:
        return False, ["未找到门店标识列"]
    expected_order = [row["store_id"] for row in expected_rows]
    actual_order = unique_sequence(stores)
    expected_growth = {row["store_id"]: row["growth_rate"] for row in expected_rows}
    growth_columns = (
        "growth_rate",
        "growth_ratio",
        "growth_pct",
        "sales_growth",
    )
    growth_column = find_metric_column(
        dataframe,
        stores,
        expected_growth,
        tolerance=0.0001,
        allow_percent_scale=True,
        preferred_columns=growth_columns,
        relative_tolerance=0.001,
    )
    actual_qualifying: list[str] = []
    if growth_column:
        growth_values = numeric_series(dataframe, growth_column[0]) / growth_column[1]
        actual_qualifying = unique_sequence(stores[growth_values > 0.10])
    messages = [
        f"返回门店={actual_order}",
        f"增长超过10%门店 actual={actual_qualifying}, expected={expected_order}",
    ]
    messages.append(f"增长率列匹配={growth_column[0] if growth_column else 'none'}")
    messages.extend(
        rate_metric_diagnostics(
            dataframe,
            stores,
            expected_growth,
            growth_column,
            growth_columns,
            relative_tolerance=0.001,
            absolute_tolerance=0.0001,
        )
    )
    return actual_qualifying == expected_order and growth_column is not None, messages


def compare_q5(
    dataframe: pd.DataFrame,
    golden: dict,
    store_names: dict[str, str],
    product_names: dict[str, str],
    conclusion: str,
) -> tuple[bool, list[str]]:
    store_rows = golden["queries"]["example_5_target_stores"]["rows"]
    sku_rows = golden["queries"]["example_5_sku_evidence"]["rows"]
    stores = entity_ids(dataframe, "store_id", "store_name", store_names)
    products = entity_ids(dataframe, "product_id", "product_name", product_names)
    if stores is None:
        return False, ["未找到门店标识列"]

    expected_stores = [row["store_id"] for row in store_rows]
    actual_stores = unique_sequence(stores)
    q1_margins = {row["store_id"]: row["q1_gross_margin_rate"] for row in store_rows}
    q2_margins = {row["store_id"]: row["q2_gross_margin_rate"] for row in store_rows}
    q1_column = find_metric_column(
        dataframe,
        stores,
        q1_margins,
        0.0001,
        allow_percent_scale=True,
        preferred_columns=(
            "q1_gross_margin_rate",
            "q1_gross_margin",
            "q1_gross_margin_pct",
            "q1_margin",
        ),
        relative_tolerance=0.001,
    )
    q2_column = find_metric_column(
        dataframe,
        stores,
        q2_margins,
        0.0001,
        allow_percent_scale=True,
        preferred_columns=(
            "q2_gross_margin_rate",
            "q2_gross_margin",
            "q2_gross_margin_pct",
            "q2_margin",
        ),
        relative_tolerance=0.001,
    )

    expected_focus_pairs: set[tuple[str, str]] = set()
    for store_id in expected_stores:
        rows = [row for row in sku_rows if row["store_id"] == store_id]
        two_lowest = sorted(rows, key=lambda row: row["q2_gross_margin_rate"])[:2]
        expected_focus_pairs.update(
            (row["store_id"], row["product_id"]) for row in two_lowest
        )
    actual_pairs = (
        set(zip(stores, products)) if products is not None else set()
    )
    focus_ok = expected_focus_pairs.issubset(actual_pairs)

    cautious = any(word in conclusion for word in ("可能", "数据显示", "迹象", "一致性"))
    causal_violation = any(
        phrase in conclusion
        for phrase in ("证明了因果", "可以证明", "确定导致", "直接导致", "就是原因")
    )
    conclusion_stores = all(
        store_id in conclusion
        or next(name for name, sid in store_names.items() if sid == store_id) in conclusion
        for store_id in expected_stores
    )
    messages = [
        f"门店 actual={actual_stores}, expected={expected_stores}",
        f"Q1毛利率列匹配={q1_column[0] if q1_column else 'none'}",
        f"Q2毛利率列匹配={q2_column[0] if q2_column else 'none'}",
        f"Golden最低毛利SKU组合均出现={focus_ok}",
        f"结论谨慎且覆盖目标门店={cautious and not causal_violation and conclusion_stores}",
    ]
    passed = (
        set(actual_stores) == set(expected_stores)
        and q1_column is not None
        and q2_column is not None
        and focus_ok
        and cautious
        and not causal_violation
        and conclusion_stores
    )
    return passed, messages


COMPARATORS: list[Callable] = [
    compare_q1,
    compare_q2,
    compare_q3,
    compare_q4,
    compare_q5,
]


def add_trace_to_report(lines: list[str], trace: dict) -> None:
    task_results = trace.get("task_results") or ()
    if task_results:
        lines.append(f"分析任务数: {len(task_results)}")
        for index, task_result in enumerate(task_results, 1):
            lines.extend(
                [
                    f"任务 {index}: {task_result.task.question}",
                    f"状态: {task_result.status}",
                    "执行 SQL:",
                    task_result.task.sql,
                    f"图表: {task_result.chart_path or 'none'}",
                ]
            )
            if task_result.status == "failed":
                lines.append(
                    f"任务错误: {task_result.error_type}: {task_result.error_message}"
                )
            elif task_result.query_result is not None:
                lines.append(
                    task_result.query_result.dataframe.to_string(
                        index=False, na_rep="NULL", max_rows=200
                    )
                )

    first_plan: SQLPlan | None = trace.get("first_plan")
    if first_plan:
        lines.extend(
            [
                "第一次 LLM SQL:",
                first_plan.sql,
                "reasoning_summary:",
                first_plan.reasoning_summary,
            ]
        )
    else:
        lines.append("第一次 LLM SQL: unavailable")

    repaired = bool(trace.get("repair_triggered"))
    lines.append(f"触发 SQL 修复: {'YES' if repaired else 'NO'}")
    if repaired:
        lines.append(f"原 SQL 错误类型: {trace.get('first_error_type')}")
        lines.append(f"原 SQL 错误: {trace.get('first_error_message')}")
        final_plan: SQLPlan | None = trace.get("final_plan")
        if final_plan:
            lines.extend(["修复后的 SQL:", final_plan.sql])

    query_result: QueryResult | None = trace.get("query_result")
    if query_result:
        lines.append("最终 DataFrame:")
        if query_result.dataframe.empty:
            lines.append("(empty)")
        else:
            lines.append(
                query_result.dataframe.to_string(index=False, na_rep="NULL", max_rows=200)
            )
        lines.append(f"结果截断: {query_result.truncated}")

    conclusion = trace.get("conclusion")
    if conclusion:
        lines.extend(["LLM 最终自然语言结论:", conclusion])


def main() -> int:
    load_dotenv(ENV_FILE, override=False)
    report = ["Text-to-SQL Online Golden Benchmark", "API Key: [NEVER LOGGED]"]

    try:
        config = load_config()
    except ConfigurationError as exc:
        configured_model = os.getenv("LLM_MODEL", "").strip()
        configured_base_url = os.getenv("LLM_BASE_URL", "").strip()
        report.extend(
            [
                "STATUS: SKIPPED",
                f"LLM_MODEL: {configured_model or 'missing'}",
                f"LLM_BASE_URL: {safe_base_url_description(configured_base_url)}",
                f"Reason: {sanitize_error(exc)}",
                "No online API request was made.",
                "",
                "SUMMARY",
                *(f"Q{index}: SKIPPED" for index in range(1, len(QUESTIONS) + 1)),
                "OVERALL: SKIPPED",
            ]
        )
        save_report(report)
        print("在线测试已停止：缺少 LLM_API_KEY 和/或 LLM_MODEL。")
        print(f"报告已保存：{REPORT_PATH}")
        return 2

    report.extend(
        [
            f"LLM_MODEL: {config.model}",
            f"LLM_BASE_URL: {safe_base_url_description(config.base_url)}",
        ]
    )
    golden = load_golden()
    store_names, product_names = dimension_maps()
    statuses: list[str] = []
    stop_remaining = False

    for index, (question, comparator, chart_expectation) in enumerate(
        zip(QUESTIONS, COMPARATORS, CHART_EXPECTATIONS),
        start=1,
    ):
        report.extend(["", "=" * 72, f"Q{index}", "用户原问题:", question])
        if stop_remaining:
            statuses.append("SKIPPED")
            report.append("STATUS: SKIPPED because an infrastructure/API error stopped further calls.")
            continue

        trace: dict = {}
        try:
            result = run_agent(question, trace=trace)
            add_trace_to_report(report, trace)
            if result.query_result is None:
                raise RuntimeError("No successful query result was returned")
            task_results = result.task_results
            chart_types = (
                [item.task.chart_type for item in task_results]
                if task_results
                else [result.plan.chart_type]
            )
            chart_paths = (
                [item.chart_path for item in task_results]
                if task_results
                else []
            )
            report.append(f"图表类型: {chart_types}")
            report.append(
                "图表路径: "
                + (", ".join(str(path) for path in chart_paths if path) or "none")
            )
            if chart_expectation == "none":
                chart_ok = all(item == "none" for item in chart_types) and not any(
                    chart_paths
                )
            elif chart_expectation == "required":
                chart_ok = any(item != "none" for item in chart_types) and any(
                    chart_paths
                )
            else:
                chart_ok = any(
                    chart_type == chart_expectation and chart_path is not None
                    for chart_type, chart_path in zip(chart_types, chart_paths)
                )
            report.append(
                f"图表策略对比: actual={chart_types}, "
                f"expected={chart_expectation}, matched={chart_ok}"
            )
            passed, comparison_messages = comparator(
                result.query_result.dataframe,
                golden,
                store_names,
                product_names,
                result.conclusion,
            )
            passed = passed and chart_ok
            report.append("Golden Result 对比:")
            report.extend(f"- {message}" for message in comparison_messages)
            status = "PASS" if passed else "FAIL"
            statuses.append(status)
            report.append(f"STATUS: {status}")
        except Exception as exc:
            add_trace_to_report(report, trace)
            category, stop_remaining = classify_error(exc, bool(config.base_url))
            report.extend(
                [
                    "图表类型: unavailable",
                    "Golden Result 对比: unavailable",
                    f"错误分类: {category}",
                    f"简洁错误: {sanitize_error(exc, config.api_key)}",
                    "STATUS: FAIL",
                ]
            )
            statuses.append("FAIL")
        save_report(report)

    report.extend(["", "=" * 72, "SUMMARY"])
    for index, status in enumerate(statuses, start=1):
        report.append(f"Q{index}: {status}")
    all_passed = statuses == ["PASS"] * len(QUESTIONS)
    report.append(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    save_report(report)

    for index, status in enumerate(statuses, start=1):
        print(f"Q{index}: {status}")
    print(f"报告已保存：{REPORT_PATH}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
