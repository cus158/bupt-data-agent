"""Lightweight, display-only evidence extraction for executed SQL queries."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from typing import Any

from .business_validator import BusinessValidationResult
from .paths import DB_PATH, KNOWLEDGE_DIR


REAL_TABLE_NAMES = (
    "sales_order",
    "store_info",
    "product_info",
    "refund_record",
)

DATE_FIELD_LABELS = {
    "order_date": "销售时间字段：sales_order.order_date",
    "refund_date": "退款时间字段：refund_record.refund_date",
}


def _read_knowledge_definitions() -> dict[str, str]:
    """Read display definitions from the existing Markdown knowledge files."""
    definitions: dict[str, str] = {}
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()

            if line.startswith("|") and line.endswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if (
                    len(cells) >= 2
                    and cells[0] not in {"业务术语", "---"}
                    and not set(cells[0]) <= {"-", ":"}
                ):
                    definitions[cells[0]] = f"{cells[0]}：{cells[1]}"

            formula = re.match(r"^-\s*`([^`=]+?)\s*=\s*([^`]+)`", line)
            if formula:
                name = formula.group(1).strip()
                definitions[name] = f"{name} = {formula.group(2).strip()}"

    return {
        name: text.replace("dim_store", "store_info").replace(
            "dim_product", "product_info"
        )
        for name, text in definitions.items()
    }


def _normalized_sql(sql: str) -> str:
    without_qualifiers = re.sub(r"\b[A-Za-z_]\w*\.", "", sql.lower())
    return re.sub(r"\s+", " ", without_qualifiers).strip()


def _compact_sql(sql: str) -> str:
    return re.sub(r"\s+", "", _normalized_sql(sql))


def _extract_tables(sql: str) -> list[str]:
    return [
        table
        for table in REAL_TABLE_NAMES
        if re.search(rf"\b{re.escape(table)}\b", sql, flags=re.IGNORECASE)
    ]


def _extract_business_terms(sql: str) -> list[str]:
    definitions = _read_knowledge_definitions()
    normalized = _normalized_sql(sql)
    compact = _compact_sql(sql)
    terms: list[str] = []

    def add(name: str) -> None:
        definition = definitions.get(name)
        if definition and definition not in terms:
            terms.append(definition)

    sales_formula = "quantity*sale_price-discount_amount" in compact
    if sales_formula or "sales_amount" in normalized:
        add("成交销额")
    if re.search(r"sum\s*\(\s*quantity\s*\)", normalized):
        terms.append("销量 = SUM(quantity)（由实际 SQL 聚合）")
    if "unit_cost" in normalized and (
        "gross_profit" in normalized or sales_formula
    ):
        add("毛利额")
    has_gross_margin = bool(
        "gross_margin" in normalized
        or "gross_margin_rate" in normalized
        or re.search(
            r"gross_profit\s*/\s*nullif\s*\([^)]*sales",
            normalized,
        )
    )
    if has_gross_margin:
        terms.append("毛利率 = SUM(毛利额) / SUM(成交销额)")
    if "refund_amount" in normalized:
        add("退损金额")
    if "refund_rate" in normalized or (
        "refund_amount" in normalized
        and (sales_formula or "sales_amount" in normalized)
        and "nullif" in normalized
    ):
        add("退损率")
    if re.search(r"channel_code\s*=\s*'o2o'", normalized):
        add("即时零售")
    if re.search(r"region\s*=\s*'[^']+'", normalized):
        add("战区")
    if "product_id" in normalized and "product_info" in normalized:
        add("SKU")
    if "growth" in normalized and ("q1" in normalized and "q2" in normalized):
        terms.append(
            "增长率 = (Q2 成交销额 - Q1 成交销额) / Q1 成交销额（由实际 SQL 计算）"
        )
    if "product_id" in normalized and (
        "quantity_growth" in normalized or "sales_growth" in normalized
    ):
        add("放量")

    return list(dict.fromkeys(terms))


def _extract_time_rules(sql: str) -> tuple[list[str], set[str], set[str]]:
    matches = re.findall(
        r"(?:\b[A-Za-z_]\w*\.)?"
        r"(order_date|refund_date)\s*(>=|<=|=|<|>)\s*"
        r"'(\d{4}-\d{2}-\d{2})'",
        sql,
        flags=re.IGNORECASE,
    )
    normalized = _normalized_sql(sql)
    fields = {field.lower() for field, _, _ in matches}
    fields.update(
        field
        for field in DATE_FIELD_LABELS
        if re.search(rf"\b{field}\b", normalized)
    )
    dates = {date for _, _, date in matches}
    rules: list[str] = []
    for field in ("order_date", "refund_date"):
        if field in fields:
            rules.append(DATE_FIELD_LABELS[field])
            predicates = [
                f"{field} {operator} '{date}'"
                for matched_field, operator, date in matches
                if matched_field.lower() == field
            ]
            rules.extend(dict.fromkeys(predicates))
            if not predicates:
                minimum, maximum = _available_date_range(field)
                if minimum and maximum:
                    label = "销售" if field == "order_date" else "退款"
                    rules.append(
                        f"相关{label}数据可用范围：{minimum} 至 {maximum}"
                    )

    if re.search(r"strftime\s*\(\s*'%y'\s*,\s*order_date\s*\)", normalized):
        rules.append("SQL 按 order_date 的自然年份分组")
    if re.search(
        r"strftime\s*\(\s*'%m'\s*,\s*order_date\s*\)"
        r"\s+between\s+'01'\s+and\s+'03'",
        normalized,
    ):
        rules.append("Q1：order_date 月份 01 至 03")
    if re.search(
        r"strftime\s*\(\s*'%m'\s*,\s*order_date\s*\)"
        r"\s+between\s+'04'\s+and\s+'06'",
        normalized,
    ):
        rules.append("Q2：order_date 月份 04 至 06")
    return list(dict.fromkeys(rules)), fields, dates


def _extract_filters(sql: str) -> list[str]:
    normalized = _normalized_sql(sql)
    sql_without_qualifiers = re.sub(r"\b[A-Za-z_]\w*\.", "", sql)
    filters: list[str] = []
    for field, value in re.findall(
        r"\b(region|channel_code|category|city|store_id|product_id|refund_reason)"
        r"\s*=\s*'([^']+)'",
        sql_without_qualifiers,
        flags=re.IGNORECASE,
    ):
        filters.append(f"{field.lower()} = '{value}'")

    for field, raw_values in re.findall(
        r"\b(region|channel_code|category|city|store_id|product_id|refund_reason)"
        r"\s+in\s*\(([^)]+)\)",
        sql_without_qualifiers,
        flags=re.IGNORECASE,
    ):
        values = re.findall(r"'((?:''|[^'])*)'", raw_values)
        if values:
            display_values = ", ".join(f"'{value}'" for value in values)
            filters.append(f"{field.lower()} IN ({display_values})")

    if "refund" in normalized and re.search(r">\s*0?\.0*5\b", normalized):
        filters.append("退损率 > 5%")
    if (
        "q1" in normalized
        and "q2" in normalized
        and re.search(r">\s*0?\.1(?:0)?\b", normalized)
    ):
        filters.append("Q2 相比 Q1 增长率 > 10%")
    if re.search(r"q2\w*sales\w*\s*>\s*q1\w*sales", normalized):
        filters.append("Q2 成交销额 > Q1 成交销额")
    if "gross_margin" in normalized and re.search(
        r"q2\w*gross_margin\w*\s*<\s*q1\w*gross_margin", normalized
    ):
        filters.append("Q2 毛利率 < Q1 毛利率")
    elif (
        "q2_gross_profit / nullif(q2_sales_amount" in normalized
        and "< q1_gross_profit / nullif(q1_sales_amount" in normalized
    ):
        filters.append("Q2 毛利率 < Q1 毛利率")
    elif (
        "q1_gross_profit / nullif(q1_sales" in normalized
        and "> q2_gross_profit / nullif(q2_sales" in normalized
    ):
        filters.append("Q2 毛利率 < Q1 毛利率")

    return list(dict.fromkeys(filters))


def _extract_aggregation(sql: str) -> list[str]:
    normalized = _normalized_sql(sql)
    group_sections = re.findall(
        r"\bgroup\s+by\s+(.+?)"
        r"(?=\bhaving\b|\border\s+by\b|\blimit\b|\bwhere\b|\bselect\b|\)|;|$)",
        normalized,
        flags=re.DOTALL,
    )
    grouped = " ".join(group_sections)
    items: list[str] = []
    if "store_id" in grouped:
        items.append("按门店聚合（store_id）")
    if "product_id" in grouped or (
        "product_id" in normalized
        and "product_info" in normalized
        and "q1" in normalized
        and "q2" in normalized
        and "sum(" in normalized
    ):
        items.append("按 SKU 聚合 / 下钻（product_id）")
    if "refund_reason" in grouped:
        items.append("按退款原因聚合（refund_reason）")
    if "category" in grouped:
        items.append("按商品类别聚合（category）")
    if "q1" in normalized and "q2" in normalized:
        items.append("按 Q1 / Q2 分期汇总")

    order_matches = re.findall(
        r"\border\s+by\s+(.+?)(?=\blimit\b|;|$)",
        normalized,
        flags=re.DOTALL,
    )
    if order_matches:
        order_clause = re.sub(r"\s+", " ", order_matches[-1]).strip()
        if len(order_clause) <= 160:
            items.append(f"排序：{order_clause}")

    limit_match = re.search(r"\blimit\s+(\d+)\b", normalized)
    if limit_match:
        items.append(f"Top-K：{limit_match.group(1)}")
    return items


def _available_date_range(field: str) -> tuple[str | None, str | None]:
    queries = {
        "order_date": "SELECT MIN(order_date), MAX(order_date) FROM sales_order",
        "refund_date": "SELECT MIN(refund_date), MAX(refund_date) FROM refund_record",
    }
    if not DB_PATH.is_file() or field not in queries:
        return None, None
    with sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True) as conn:
        return conn.execute(queries[field]).fetchone()


def _unique_data_year(fields: set[str]) -> str | None:
    if not fields:
        return None
    years: set[str] = set()
    for field in fields:
        minimum, maximum = _available_date_range(field)
        if not minimum or not maximum or minimum[:4] != maximum[:4]:
            return None
        years.add(minimum[:4])
    return next(iter(years)) if len(years) == 1 else None


def _temporal_disambiguation_note(
    question: str,
    sql: str,
    fields: set[str],
    sql_dates: set[str],
) -> str | None:
    has_quarter = bool(
        re.search(r"Q[1-4]|第[一二三四1234]季度", question, flags=re.IGNORECASE)
    )
    has_explicit_year = bool(re.search(r"(?:19|20)\d{2}", question))
    if not has_quarter or has_explicit_year:
        return None

    unique_year = _unique_data_year(fields)
    sql_years = {date[:4] for date in sql_dates}
    if unique_year and sql_years == {unique_year}:
        return (
            "Temporal Context 自动消歧：用户未明确指定年份；"
            f"相关数据当前只覆盖 {unique_year} 年，SQL 按 {unique_year} 年解释。"
        )
    if unique_year and not sql_years and re.search(
        r"strftime\s*\(\s*'%Y'\s*,\s*(?:[A-Za-z_]\w*\.)?order_date\s*\)",
        sql,
        flags=re.IGNORECASE,
    ):
        return (
            "Temporal Context 自动消歧：用户未明确指定年份；"
            f"相关数据当前只覆盖 {unique_year} 年，"
            f"本次按年份分组的 Q1/Q2 结果对应 {unique_year} 年。"
        )
    return None


def build_query_evidence(
    sql: str,
    question: str,
    business_validation: BusinessValidationResult | None = None,
) -> dict[str, list[str]]:
    """Build auditable display metadata from the SQL that actually ran."""
    time_rules, date_fields, sql_dates = _extract_time_rules(sql)
    notes: list[str] = []
    temporal_note = _temporal_disambiguation_note(
        question,
        sql,
        date_fields,
        sql_dates,
    )
    if temporal_note:
        notes.append(temporal_note)

    normalized = _normalized_sql(sql)
    has_gross_margin = bool(
        "gross_margin" in normalized
        or re.search(
            r"gross_profit\s*/\s*nullif\s*\([^)]*sales",
            normalized,
        )
    )
    if (
        "product_id" in normalized
        and has_gross_margin
        and ("q1" in normalized and "q2" in normalized)
    ):
        notes.append(
            "SKU 下钻用于提供相关性证据；结果只能说明可能存在毛利拖累迹象，"
            "不能证明因果关系。"
        )

    status = ["SQL 安全检查已通过"]
    if business_validation and business_validation.valid:
        status.append("Business Rule Validator 已通过")
    status.extend(
        [
            "SQLite 只读查询已执行",
            "结果来自实际数据库查询",
        ]
    )

    return {
        "tables": _extract_tables(sql),
        "business_terms": _extract_business_terms(sql),
        "time_rules": time_rules,
        "filters": _extract_filters(sql),
        "aggregation": _extract_aggregation(sql),
        "notes": notes,
        "business_warnings": [
            issue.message
            for issue in business_validation.warnings
        ] if business_validation else [],
        "status": status,
    }


def build_analysis_evidence(
    tasks: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build task-scoped evidence without merging unrelated SQL statements.

    Each input item must provide task_id, question, and sql; an optional
    business_validation value is forwarded to the existing single-query extractor.
    """
    results: list[dict[str, Any]] = []
    for item in tasks:
        task_id = str(item.get("task_id") or "").strip()
        question = str(item.get("question") or "").strip()
        sql = str(item.get("sql") or "").strip()
        if not task_id or not question or not sql:
            raise ValueError("Each evidence task requires task_id, question, and sql")
        results.append(
            {
                "task_id": task_id,
                "question": question,
                "sql": sql,
                "evidence": build_query_evidence(
                    sql,
                    question,
                    item.get("business_validation"),
                ),
            }
        )
    return results
