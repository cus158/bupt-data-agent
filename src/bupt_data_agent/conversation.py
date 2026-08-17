"""Lightweight, session-scoped conversation context for follow-up data questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


MAX_ENTITIES_PER_TYPE = 10

SINGULAR_REFERENCE_PATTERN = re.compile(
    r"它(?!们)|这个(?:门店|SKU|商品)?|那个(?:门店|SKU|商品)?|"
    r"刚才那个|这家(?:门店)?|那家(?:门店)?",
    flags=re.IGNORECASE,
)
PLURAL_REFERENCE_PATTERN = re.compile(
    r"它们|这些(?:门店|SKU|商品)?|那些(?:门店|SKU|商品)?|"
    r"刚才这些|刚才那些",
    flags=re.IGNORECASE,
)
FOLLOW_UP_PATTERN = re.compile(r"刚才|上一轮|前面(?:的)?|继续看|再看", re.IGNORECASE)


@dataclass(frozen=True)
class ContextResolution:
    use_context: bool
    reference_mode: str | None
    prompt_context: str | None
    clarification_question: str | None


def reference_mode(question: str) -> str | None:
    if PLURAL_REFERENCE_PATTERN.search(question):
        return "plural"
    if SINGULAR_REFERENCE_PATTERN.search(question):
        return "singular"
    if FOLLOW_UP_PATTERN.search(question):
        return "follow_up"
    return None


def _extract_entities(
    dataframe: pd.DataFrame,
    id_column: str,
    name_column: str,
) -> tuple[list[dict[str, str]], int, bool]:
    if id_column not in dataframe.columns:
        return [], 0, False
    columns = [id_column]
    if name_column in dataframe.columns:
        columns.append(name_column)
    entity_frame = dataframe[columns].dropna(subset=[id_column]).copy()
    entity_frame[id_column] = entity_frame[id_column].astype(str)
    entity_frame = entity_frame.drop_duplicates(subset=[id_column], keep="first")
    count = len(entity_frame)
    items: list[dict[str, str]] = []
    for _, row in entity_frame.head(MAX_ENTITIES_PER_TYPE).iterrows():
        item = {id_column: str(row[id_column])}
        if name_column in entity_frame.columns and pd.notna(row[name_column]):
            item[name_column] = str(row[name_column])
        items.append(item)
    return items, count, count > MAX_ENTITIES_PER_TYPE


def _extract_time_context(question: str, sql: str, dataframe: pd.DataFrame) -> dict[str, Any]:
    years = set(re.findall(r"(?:19|20)\d{2}", question))
    years.update(re.findall(r"'(\d{4})-\d{2}-\d{2}'", sql))
    for column in ("sales_year", "year"):
        if column in dataframe.columns:
            values = dataframe[column].dropna().astype(str).str[:4].unique().tolist()
            years.update(value for value in values if re.fullmatch(r"\d{4}", value))

    period: str | None = None
    has_q1 = bool(re.search(r"Q1|第一季度", question, re.IGNORECASE))
    has_q2 = bool(re.search(r"Q2|第二季度", question, re.IGNORECASE))
    if has_q1 and has_q2:
        period = "Q1-Q2"
    elif has_q1:
        period = "Q1"
    elif has_q2:
        period = "Q2"
    elif "上半年" in question:
        period = "H1"
    elif "下半年" in question:
        period = "H2"
    else:
        month_days = set(re.findall(r"'\d{4}-(\d{2}-\d{2})'", sql))
        if {"01-01", "07-01"}.issubset(month_days):
            period = "H1"
        elif {"04-01", "07-01"}.issubset(month_days):
            period = "Q2"
        elif {"01-01", "04-01"}.issubset(month_days):
            period = "Q1"

    return {
        "year": int(next(iter(years))) if len(years) == 1 else None,
        "period": period,
    }


def _extract_metrics(question: str, sql: str) -> list[str]:
    combined = f"{question}\n{sql}".lower()
    metrics: list[str] = []
    candidates = (
        ("退损率", ("退损率", "refund_rate", "refund_loss_rate")),
        ("退款金额", ("退款金额", "refund_amount")),
        ("毛利率", ("毛利率", "gross_margin", "margin_rate")),
        ("毛利额", ("毛利额", "gross_profit")),
        ("成交销额", ("成交销额", "销额", "sales_amount", "quantity *", "quantity*")),
        ("销量", ("销量", "sales_quantity", "total_quantity", "sum(quantity")),
    )
    for label, markers in candidates:
        if any(marker in combined for marker in markers):
            metrics.append(label)
    return metrics


def extract_turn_context(
    question: str,
    sql: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Extract bounded context only from the actual executed result and SQL."""
    stores, store_count, stores_truncated = _extract_entities(
        dataframe, "store_id", "store_name"
    )
    products, product_count, products_truncated = _extract_entities(
        dataframe, "product_id", "product_name"
    )
    time_context = _extract_time_context(question, sql, dataframe)
    metrics = _extract_metrics(question, sql)

    summary_parts: list[str] = []
    if stores:
        labels = [
            " ".join(filter(None, (item.get("store_id"), item.get("store_name"))))
            for item in stores[:3]
        ]
        summary_parts.append("门店：" + "、".join(labels))
    if products:
        labels = [
            " ".join(filter(None, (item.get("product_id"), item.get("product_name"))))
            for item in products[:3]
        ]
        summary_parts.append("SKU：" + "、".join(labels))
    if metrics:
        summary_parts.append("指标：" + "、".join(metrics))
    if time_context["year"] or time_context["period"]:
        summary_parts.append(
            "时间："
            + " ".join(
                value
                for value in (
                    str(time_context["year"]) if time_context["year"] else None,
                    time_context["period"],
                )
                if value
            )
        )

    return {
        "question": question,
        "entities": {"stores": stores, "products": products},
        "entity_counts": {"stores": store_count, "products": product_count},
        "entities_truncated": {
            "stores": stores_truncated,
            "products": products_truncated,
        },
        "time_context": time_context,
        "metrics": metrics,
        "result_summary": "；".join(summary_parts) or "本轮查询未返回可复用实体。",
    }


def format_conversation_context(context: dict[str, Any]) -> str:
    """Format a compact, bounded context block for the existing SQL-generation call."""
    compact = {
        "previous_question": context.get("question"),
        "stores": context.get("entities", {}).get("stores", [])[:MAX_ENTITIES_PER_TYPE],
        "products": context.get("entities", {}).get("products", [])[:MAX_ENTITIES_PER_TYPE],
        "entity_counts": context.get("entity_counts", {}),
        "entities_truncated": context.get("entities_truncated", {}),
        "time": context.get("time_context", {}),
        "metrics": context.get("metrics", []),
        "previous_result_summary": context.get("result_summary"),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _candidate_entity_types(question: str, context: dict[str, Any]) -> list[str]:
    if re.search(r"门店|店铺", question):
        return ["stores"]
    if re.search(r"SKU|商品|产品", question, re.IGNORECASE):
        return ["products"]
    entities = context.get("entities", {})
    return [name for name in ("stores", "products") if entities.get(name)]


def _entity_options(context: dict[str, Any], entity_type: str) -> list[str]:
    id_column, name_column = (
        ("store_id", "store_name") if entity_type == "stores" else ("product_id", "product_name")
    )
    return [
        " ".join(filter(None, (item.get(id_column), item.get(name_column))))
        for item in context.get("entities", {}).get(entity_type, [])[:3]
    ]


def resolve_conversation_context(
    question: str,
    context: dict[str, Any] | None,
) -> ContextResolution:
    """Decide whether context is relevant and whether a reference is resolvable."""
    mode = reference_mode(question)
    if mode is None:
        return ContextResolution(False, None, None, None)
    if not context:
        return ContextResolution(
            False,
            mode,
            None,
            "当前没有可用的上一轮查询上下文，请明确你指的是哪个门店或SKU。",
        )

    candidate_types = _candidate_entity_types(question, context)
    if not candidate_types:
        return ContextResolution(
            False,
            mode,
            None,
            "上一轮结果没有可用于解析该指代的门店或SKU，请明确查询对象。",
        )

    explicit_store_ids = re.findall(r"\bS\d+\b", question, flags=re.IGNORECASE)
    explicit_product_ids = re.findall(r"\bP\d+\b", question, flags=re.IGNORECASE)
    if mode == "singular" and len(explicit_store_ids) + len(explicit_product_ids) == 1:
        return ContextResolution(
            True,
            mode,
            format_conversation_context(context),
            None,
        )

    if mode == "singular":
        if len(candidate_types) > 1:
            return ContextResolution(
                False,
                mode,
                None,
                "上一轮同时包含门店和SKU，请明确你指的是哪个对象。",
            )
        entity_type = candidate_types[0]
        count = int(context.get("entity_counts", {}).get(entity_type, 0))
        if count != 1:
            options = _entity_options(context, entity_type)
            label = "门店" if entity_type == "stores" else "SKU"
            option_text = "、".join(options)
            if count > len(options):
                option_text += "等"
            return ContextResolution(
                False,
                mode,
                None,
                f"上一轮返回了{count}个{label}，请明确你指的是{option_text}中的哪一个。",
            )

    return ContextResolution(
        True,
        mode,
        format_conversation_context(context),
        None,
    )


def context_display_summary(context: dict[str, Any] | None) -> str | None:
    if not context:
        return None
    return str(context.get("result_summary") or "").strip() or None
