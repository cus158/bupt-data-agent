"""Deterministic, conservative validation of high-confidence business SQL rules."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BusinessRuleIssue:
    rule: str
    message: str


@dataclass(frozen=True)
class BusinessValidationResult:
    valid: bool
    violations: tuple[BusinessRuleIssue, ...]
    warnings: tuple[BusinessRuleIssue, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "violations": [issue.__dict__.copy() for issue in self.violations],
            "warnings": [issue.__dict__.copy() for issue in self.warnings],
        }


def _normalized_sql(sql: str) -> str:
    without_literals = re.sub(r"'(?:''|[^'])*'", "''", sql.lower())
    without_qualifiers = re.sub(r"\b[a-z_]\w*\.", "", without_literals)
    return re.sub(r"\s+", " ", without_qualifiers).strip()


def _compact_sql(sql: str) -> str:
    return re.sub(r"\s+", "", _normalized_sql(sql))


def _function_arguments(compact_sql: str, function_name: str) -> list[str]:
    """Extract balanced function arguments without attempting full SQL parsing."""
    arguments: list[str] = []
    pattern = re.compile(rf"\b{re.escape(function_name.lower())}\(")
    for match in pattern.finditer(compact_sql):
        start = match.end()
        depth = 1
        index = start
        while index < len(compact_sql) and depth:
            if compact_sql[index] == "(":
                depth += 1
            elif compact_sql[index] == ")":
                depth -= 1
            index += 1
        if depth == 0:
            arguments.append(compact_sql[start : index - 1])
    return arguments


def _select_segments(normalized_sql: str) -> list[str]:
    positions = [match.start() for match in re.finditer(r"\bselect\b", normalized_sql)]
    return [
        normalized_sql[start : positions[index + 1] if index + 1 < len(positions) else None]
        for index, start in enumerate(positions)
    ]


def _has_date_request(question: str) -> bool:
    return bool(
        re.search(
            r"(?:19|20)\d{2}|Q[1-4]|第[一二三四1234]季度|"
            r"上半年|下半年|\d{1,2}月|期间|时间段",
            question,
            flags=re.IGNORECASE,
        )
    )


def _has_refund_date_filter(normalized_sql: str) -> bool:
    return bool(
        re.search(
            r"\brefund_date\b\s*(?:>=|<=|=|>|<|between\b|in\s*\()",
            normalized_sql,
        )
        or re.search(
            r"(?:date|strftime)\s*\([^)]*\brefund_date\b",
            normalized_sql,
        )
    )


def _has_zero_division_guard(normalized_sql: str) -> bool:
    return "nullif(" in normalized_sql or bool(
        re.search(
            r"\bcase\b.*?\bwhen\b.*?(?:=|is)\s*0\b.*?\bthen\b",
            normalized_sql,
            flags=re.DOTALL,
        )
    )


def validate_business_rules(
    question: str,
    sql: str,
    business_context: str,
) -> BusinessValidationResult:
    """Validate only clear business-rule violations; warnings never block execution."""
    normalized = _normalized_sql(sql)
    compact = _compact_sql(sql)
    context = business_context.lower()
    violations: list[BusinessRuleIssue] = []
    warnings: list[BusinessRuleIssue] = []

    def violate(rule: str, message: str) -> None:
        if not any(issue.rule == rule for issue in violations):
            violations.append(BusinessRuleIssue(rule, message))

    def warn(rule: str, message: str) -> None:
        if not any(issue.rule == rule for issue in warnings):
            warnings.append(BusinessRuleIssue(rule, message))

    # Rule 7: historical Markdown names are never executable table names.
    wrong_tables = [
        table for table in ("dim_store", "dim_product") if re.search(rf"\b{table}\b", normalized)
    ]
    if wrong_tables:
        replacements = {"dim_store": "store_info", "dim_product": "product_info"}
        details = ", ".join(
            f"{table} 应改为 {replacements[table]}" for table in wrong_tables
        )
        violate("real_table_names", f"SQL 使用了历史表名：{details}。")

    asks_sales_amount = "销额" in question or "销售额" in question
    asks_quantity = "销量" in question or "销售数量" in question
    sales_formula_defined = all(
        token in context for token in ("quantity", "sale_price", "discount_amount")
    )

    # Rule 1: only apply when the user explicitly asks for the sales-amount metric.
    if (
        asks_sales_amount
        and sales_formula_defined
        and "quantity*sale_price" in compact
        and "discount_amount" not in compact
    ):
        violate(
            "transaction_sales_amount",
            "成交销额必须按 quantity * sale_price - discount_amount 计算；当前公式未扣减优惠金额。",
        )

    # Rule 8: an explicit user metric overrides a business-term default.
    has_sum_quantity = "sum(quantity)" in compact or bool(
        re.search(r"sum\(case.+?quantity.+?end\)", compact)
    )
    has_sales_formula = "quantity*sale_price" in compact
    if asks_quantity and has_sales_formula and "sum(" in compact and not has_sum_quantity:
        violate(
            "explicit_metric_priority",
            "用户明确要求按销量分析，SQL 却只聚合了金额指标。",
        )
    if asks_sales_amount and has_sum_quantity and not has_sales_formula:
        violate(
            "explicit_metric_priority",
            "用户明确要求成交销额，SQL 却只聚合了销量。",
        )

    # Rule 2: reject only a clearly row-level margin expression inside AVG(...).
    if "毛利率" in question and "毛利率" in context:
        for argument in _function_arguments(compact, "avg"):
            if "/" in argument and any(
                marker in argument for marker in ("unit_cost", "gross_profit", "margin")
            ):
                violate(
                    "gross_margin_aggregation",
                    "汇总毛利率不能使用 AVG(单笔毛利率)，应按总毛利额 / 总成交销额计算。",
                )
                break

    refund_terms = ("退款", "退损")
    asks_refund = any(term in question for term in refund_terms)

    # Rule 3: an explicitly requested refund period must filter refund_date.
    if (
        asks_refund
        and _has_date_request(question)
        and "refund_date" in context
        and "refund_record" in normalized
        and not _has_refund_date_filter(normalized)
    ):
        violate(
            "refund_time_field",
            "退款发生期间必须使用 refund_record.refund_date 过滤，不能只使用销售订单日期。",
        )

    # Rule 4: refund loss rate is an amount ratio, not a count ratio.
    if "退损率" in question and "退损率" in context:
        count_ratio = bool(
            re.search(
                r"count\([^)]*refund_id[^)]*\)\*?(?:1(?:\.0)?)?/"
                r"count\([^)]*order_id[^)]*\)",
                compact,
            )
            or re.search(r"refund_count/(?:nullif\()?order_count", compact)
        )
        if count_ratio:
            violate(
                "refund_loss_rate_formula",
                "退损率必须使用退款金额 / 成交销额，不能使用退款次数 / 订单数。",
            )

    # Rule 5: warn only when one SELECT block visibly sums both raw detail measures.
    if "退损率" in question or (asks_refund and asks_sales_amount):
        for segment in _select_segments(normalized):
            segment_compact = re.sub(r"\s+", "", segment)
            if (
                "refund_record" in segment
                and "sales_order" in segment
                and "join" in segment
                and "sum(" in segment_compact
                and "refund_amount" in segment_compact
                and "quantity*sale_price" in segment_compact
            ):
                warn(
                    "refund_sales_join_amplification",
                    "退款与销售明细在同一层直接汇总，未来一单多退款时可能放大销售额；建议分别聚合后再 JOIN。",
                )
                break

    # Rule 6: engineering warning only; it never makes an otherwise valid SQL invalid.
    rate_question = any(
        term in question for term in ("增长率", "增长", "毛利率", "退损率", "占比", "比例")
    )
    sql_without_literals = re.sub(r"'(?:''|[^'])*'", "''", normalized)
    if rate_question and "/" in sql_without_literals and not _has_zero_division_guard(normalized):
        warn(
            "division_by_zero",
            "比例或增长率计算未显式使用 NULLIF 或 CASE 保护零分母。",
        )

    return BusinessValidationResult(
        valid=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
    )

