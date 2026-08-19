"""Minimal Text-to-SQL agent for the BUPT assessment dataset."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .business_validator import BusinessValidationResult, validate_business_rules
from .conversation import extract_turn_context, resolve_conversation_context
from .paths import DB_PATH, ENV_FILE, KNOWLEDGE_DIR, OUTPUTS_DIR


REAL_TABLE_NAMES = ("store_info", "product_info", "sales_order", "refund_record")
CHART_TYPES = {"bar", "line", "pie", "none"}
MAX_RESULT_ROWS = 200
LLM_RESULT_ROWS = 100
QUERY_TIMEOUT_SECONDS = 5.0

FORBIDDEN_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
}


class AgentError(RuntimeError):
    """Base class for expected agent errors."""


class ConfigurationError(AgentError):
    """Raised when required environment configuration is missing."""


class LLMResponseError(AgentError):
    """Raised when the LLM response is missing or malformed."""


class SQLSafetyError(AgentError):
    """Raised when generated SQL does not pass the local safety policy."""


class BusinessRuleValidationError(AgentError):
    """Raised when SQL clearly violates a deterministic business rule."""

    def __init__(self, result: BusinessValidationResult):
        self.result = result
        details = "; ".join(
            f"[{issue.rule}] {issue.message}" for issue in result.violations
        )
        super().__init__(f"Business Rule Validator rejected the SQL: {details}")


class SQLExecutionError(AgentError):
    """Raised when SQLite cannot execute an otherwise safe query."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class SQLPlan:
    sql: str | None
    reasoning_summary: str
    chart_type: str
    status: str = "ready"
    clarification_question: str | None = None
    ambiguity_type: str | None = None


@dataclass(frozen=True)
class QueryResult:
    dataframe: pd.DataFrame
    truncated: bool


@dataclass(frozen=True)
class AgentResult:
    question: str
    first_plan: SQLPlan
    plan: SQLPlan
    query_result: QueryResult | None
    conclusion: str | None
    repair_triggered: bool
    first_error_type: str | None
    first_error_message: str | None
    business_validation: BusinessValidationResult | None
    turn_context: dict[str, Any] | None
    conversation_context_used: bool


def load_config() -> LLMConfig:
    load_dotenv(ENV_FILE, override=False)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    missing = []
    if not api_key:
        missing.append("LLM_API_KEY")
    if not model:
        missing.append("LLM_MODEL")
    if missing:
        raise ConfigurationError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return LLMConfig(api_key=api_key, model=model, base_url=base_url)


def create_llm_client(config: LLMConfig) -> OpenAI:
    kwargs: dict[str, Any] = {
        "api_key": config.api_key,
        "timeout": 60.0,
        "max_retries": 1,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def load_business_context() -> str:
    if not KNOWLEDGE_DIR.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {KNOWLEDGE_DIR}")

    paths = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown files found in: {KNOWLEDGE_DIR}")

    sections = []
    for path in paths:
        content = path.read_text(encoding="utf-8-sig").strip()
        sections.append(f"===== {path.name} =====\n{content}")
    return "\n\n".join(sections)


def _connect_read_only() -> sqlite3.Connection:
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. "
            "Run 'python -m bupt_data_agent.prepare_db' first."
        )
    conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_schema_context() -> str:
    sections = []
    with _connect_read_only() as conn:
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = set(REAL_TABLE_NAMES) - existing_tables
        if missing:
            raise RuntimeError(f"Database is missing required tables: {sorted(missing)}")

        for table_name in REAL_TABLE_NAMES:
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            foreign_keys = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
            lines = [f"TABLE {table_name}"]
            for _, name, data_type, not_null, default_value, primary_key in columns:
                attributes = [data_type]
                if not_null:
                    attributes.append("NOT NULL")
                if primary_key:
                    attributes.append("PRIMARY KEY")
                if default_value is not None:
                    attributes.append(f"DEFAULT {default_value}")
                lines.append(f"- {name}: {' '.join(attributes)}")
            for foreign_key in foreign_keys:
                _, _, target_table, source_column, target_column, *_ = foreign_key
                lines.append(
                    f"- FOREIGN KEY {source_column} -> "
                    f"{target_table}.{target_column}"
                )
            sections.append("\n".join(lines))

        sales_min, sales_max = conn.execute(
            "SELECT MIN(order_date), MAX(order_date) FROM sales_order"
        ).fetchone()
        refund_min, refund_max = conn.execute(
            "SELECT MIN(refund_date), MAX(refund_date) FROM refund_record"
        ).fetchone()

    sections.append(
        "DATA TEMPORAL CONTEXT\n"
        "sales_order.order_date available range:\n"
        f"{sales_min or 'no data'} to {sales_max or 'no data'}\n\n"
        "refund_record.refund_date available range:\n"
        f"{refund_min or 'no data'} to {refund_max or 'no data'}"
    )

    sections.append(
        "IMPORTANT TABLE-NAME CORRECTION:\n"
        "business_terms.md mentions dim_store and dim_product, but those tables do not "
        "exist. SQL must use store_info and product_info."
    )
    return "\n\n".join(sections)


def _sql_system_prompt(
    schema_context: str,
    business_context: str,
    conversation_context: str | None = None,
) -> str:
    conversation_section = (
        "\n\nCONVERSATION CONTEXT (bounded structured data from the previous actual "
        f"query result):\n{conversation_context}"
        if conversation_context
        else ""
    )
    return f"""You are a careful Text-to-SQL analyst for a small SQLite database.

Return exactly one JSON object with exactly these keys:
{{
  "status": "ready|needs_clarification",
  "clarification_question": null,
  "ambiguity_type": null,
  "sql": "one SQLite read-only query",
  "reasoning_summary": "a short summary of tables, metric and filters used",
  "chart_type": "bar|line|pie|none"
}}

Do not output Markdown fences. Do not reveal private chain-of-thought. The
reasoning_summary must be at most three short sentences and only explain which tables,
business metric, date range and filters were used.

Intent-completeness rules (apply before writing SQL):
1. First decide whether the question has enough information for one reliable business
   query. Business knowledge in the supplied Markdown has the highest semantic
   priority. If it defines a term or its ranking convention, apply that definition and
   do not ask about a generic-language ambiguity. In particular, do not second-guess a
   documented business term merely because words such as "best" appear in it. Treat the
   documented 动销 ranking convention as defined: rank "动销最好" by transaction sales
   amount unless the user explicitly names another metric such as sales quantity; do
   not ask the user to redefine that term.
2. Set status="ready" when business knowledge or database context can uniquely resolve
   the meaning. Then clarification_question and ambiguity_type must be null, and sql
   must contain the read-only query.
3. If two or more equally reasonable interpretations would materially change the SQL
   or conclusion, set status="needs_clarification". Then sql must be null,
   chart_type="none", ambiguity_type must briefly classify the ambiguity (for example
   "metric", "time", "object", or "scope"), and clarification_question must ask one
   concise, specific question in Chinese. Offer at most three common options and do not
   turn the question into a long checklist.
4. Vague words such as "表现" or an undocumented use of "最好" require clarification
   when the user supplies no metric and sales quantity, transaction sales amount,
   gross profit, gross margin, or refund loss could all reasonably answer it. For a
   product/SKU ranking, ask whether to use sales quantity or transaction sales amount.
   For a store or region, ask about no more than transaction sales amount, gross margin,
   and refund loss rate. Do not add a fourth metric option.
5. A missing Top-K count is not by itself ambiguous: an ordered result is acceptable.
   A missing time range is also not automatically ambiguous; use the relevant available
   data range and mention that basis. Quarter expressions follow the temporal rules
   below.
6. Ask only the single most blocking clarification question. Never emit guessed SQL
   together with needs_clarification.

Conversation-context rules:
1. The current user question always has higher priority than previous context.
2. Use CONVERSATION CONTEXT only to resolve an explicit follow-up, pronoun, or omitted
   reference in the current question. Never inherit an unrelated store, SKU, period,
   channel, region, metric, threshold, or filter into an independent question.
3. Explicit entities, dates, periods, and metrics in the current question replace any
   conflicting values in previous context.
4. A singular reference may inherit exactly one matching entity. If multiple matching
   entities remain, return needs_clarification and do not choose the first result.
5. A plural reference may inherit the complete bounded entity set from context.
6. Context is evidence from the previous actual query result, not permission to bypass
   schema, SQL safety, business rules, or temporal rules.

Mandatory SQL rules:
1. Use only tables and columns present in the supplied SQLite schema.
2. Follow the Markdown business definitions exactly.
3. Prefer half-open date ranges: date >= start AND date < end.
4. Transaction sales amount = quantity * sale_price - discount_amount.
5. Gross profit = sales amount - quantity * unit_cost.
6. Aggregate gross margin = SUM(gross profit) / SUM(sales amount). Never use AVG of
   row-level gross margin.
7. Sales periods use order_date. Refund periods use refund_date.
8. When sales and refunds need separate time filters or aggregation, aggregate them in
   separate CTEs before joining so detail joins cannot duplicate amounts.
9. SQL must be compatible with SQLite and must use NULLIF for possible zero divisors.
10. Only produce a SELECT query or a legal WITH ... SELECT query. Never modify schema
    or data and never use PRAGMA, ATTACH, or database administration statements.
11. Do not invent a numeric threshold for an undefined concept such as low margin or
    volume expansion. Return the underlying evidence instead.
12. If the question has both a store-level filter and a reason/SKU drill-down, use CTEs
    and return one result table containing the store metrics alongside the detail rows.
13. For low-margin SKU / margin-drag questions, provide evidence only. Do not claim
    causality.
14. First determine whether the current user explicitly requests a chart or
    visualization. If not, chart_type must be none, even when the result would be easy
    to visualize. If the user explicitly requests bar, line, or pie, use that type. If
    the user asks for a chart/visualization but does not specify a type, use bar for
    rankings or category comparisons, line for continuous time series, pie only for an
    explicit small part-to-whole result, and none when no meaningful chart can be made.
15. A threshold such as growth > 10% or refund rate > 5% applies only when the current
    user question explicitly requests it. It is never a default rule for other growth
    or refund questions.
16. Respect any year explicitly stated by the user. If the user mentions Q1, Q2,
    first quarter, second quarter, or another quarter without a year, use DATA TEMPORAL
    CONTEXT to resolve the year for the relevant date column.
17. If the relevant business data covers exactly one calendar year, an unspecified
    quarter means that unique year. Build the quarter with half-open boundaries. For
    example, Q1 is January 1 inclusive to April 1 exclusive, and Q2 is April 1
    inclusive to July 1 exclusive.
18. If the relevant data spans multiple calendar years and the user did not specify a
    year, never guess a year. Return status=needs_clarification with sql=null and ask
    the user to specify the year.

Actual SQLite schema:
{schema_context}

Business knowledge from all Markdown files:
{business_context}{conversation_section}
"""


def _parse_sql_plan(payload: Any) -> SQLPlan:
    if not isinstance(payload, dict):
        raise LLMResponseError("LLM response must be a JSON object")
    required_keys = {
        "status",
        "clarification_question",
        "ambiguity_type",
        "sql",
        "reasoning_summary",
        "chart_type",
    }
    if set(payload) != required_keys:
        raise LLMResponseError(
            f"LLM JSON keys must be exactly {sorted(required_keys)}; "
            f"received {sorted(payload)}"
        )

    status = payload["status"]
    clarification_question = payload["clarification_question"]
    ambiguity_type = payload["ambiguity_type"]
    sql = payload["sql"]
    reasoning_summary = payload["reasoning_summary"]
    chart_type = payload["chart_type"]
    if status not in {"ready", "needs_clarification"}:
        raise LLMResponseError(f"Invalid status: {status!r}")
    if not isinstance(reasoning_summary, str) or not reasoning_summary.strip():
        raise LLMResponseError("LLM response contains an invalid reasoning_summary")
    if not isinstance(chart_type, str) or chart_type.lower() not in CHART_TYPES:
        raise LLMResponseError(f"Invalid chart_type: {chart_type!r}")

    if status == "ready":
        if not isinstance(sql, str) or not sql.strip():
            raise LLMResponseError("A ready plan must contain a non-empty SQL value")
        if clarification_question is not None or ambiguity_type is not None:
            raise LLMResponseError(
                "A ready plan must use null clarification_question and ambiguity_type"
            )
    else:
        if sql is not None:
            raise LLMResponseError("A clarification plan must use sql=null")
        if chart_type.lower() != "none":
            raise LLMResponseError("A clarification plan must use chart_type=none")
        if not isinstance(clarification_question, str) or not clarification_question.strip():
            raise LLMResponseError(
                "A clarification plan must contain a clarification_question"
            )
        if not isinstance(ambiguity_type, str) or not ambiguity_type.strip():
            raise LLMResponseError("A clarification plan must contain an ambiguity_type")

    return SQLPlan(
        sql=sql.strip() if isinstance(sql, str) else None,
        reasoning_summary=reasoning_summary.strip(),
        chart_type=chart_type.lower(),
        status=status,
        clarification_question=(
            clarification_question.strip()
            if isinstance(clarification_question, str)
            else None
        ),
        ambiguity_type=ambiguity_type.strip() if isinstance(ambiguity_type, str) else None,
    )


def _call_json_plan(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> SQLPlan:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {exc}") from exc
    return _parse_sql_plan(payload)


def _apply_chart_policy(
    question: str,
    proposed_chart_type: str,
    status: str,
) -> str:
    """Apply the user's explicit chart intent to the model's proposed chart type."""
    if status != "ready":
        return "none"

    explicit_types = (
        (r"柱状图|柱形图|条形图|\bbar\s+chart\b", "bar"),
        (r"折线图|\bline\s+chart\b", "line"),
        (r"饼图|饼状图|\bpie\s+chart\b", "pie"),
    )
    for pattern, chart_type in explicit_types:
        if re.search(pattern, question, re.IGNORECASE):
            return chart_type

    chart_requested = bool(
        re.search(
            r"画图|绘图|图表|可视化|生成.{0,12}图|绘制.{0,12}图|画.{0,12}图|"
            r"对比图|\bchart\b|\bvisuali[sz](?:e|ation)\b",
            question,
            re.IGNORECASE,
        )
    )
    return proposed_chart_type if chart_requested else "none"


def generate_sql_plan(
    client: OpenAI,
    model: str,
    question: str,
    schema_context: str,
    business_context: str,
    conversation_context: str | None = None,
) -> SQLPlan:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    plan = _call_json_plan(
        client,
        model,
        _sql_system_prompt(schema_context, business_context, conversation_context),
        question.strip(),
    )
    return SQLPlan(
        sql=plan.sql,
        reasoning_summary=plan.reasoning_summary,
        chart_type=_apply_chart_policy(question, plan.chart_type, plan.status),
        status=plan.status,
        clarification_question=plan.clarification_question,
        ambiguity_type=plan.ambiguity_type,
    )


def repair_sql_plan(
    client: OpenAI,
    model: str,
    question: str,
    failed_plan: SQLPlan,
    error_message: str,
    schema_context: str,
    business_context: str,
    conversation_context: str | None = None,
) -> SQLPlan:
    user_prompt = f"""The previous SQL for the question failed SQL safety validation,
business-rule validation, or SQLite execution.

Original question:
{question}

Previous SQL:
{failed_plan.sql}

Error:
{error_message}

Return one corrected JSON object. Keep the original analytical intent and obey all
system rules. Do not explain the error outside reasoning_summary."""
    plan = _call_json_plan(
        client,
        model,
        _sql_system_prompt(schema_context, business_context, conversation_context),
        user_prompt,
    )
    return SQLPlan(
        sql=plan.sql,
        reasoning_summary=plan.reasoning_summary,
        chart_type=_apply_chart_policy(question, plan.chart_type, plan.status),
        status=plan.status,
        clarification_question=plan.clarification_question,
        ambiguity_type=plan.ambiguity_type,
    )


def _mask_single_quoted_literals(sql: str) -> str:
    output = []
    index = 0
    in_string = False
    while index < len(sql):
        char = sql[index]
        if in_string:
            if char == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                output.extend((" ", " "))
                index += 2
                continue
            if char == "'":
                in_string = False
            output.append(" ")
        else:
            if char == "'":
                in_string = True
                output.append(" ")
            else:
                output.append(char)
        index += 1
    if in_string:
        raise SQLSafetyError("SQL contains an unterminated string literal")
    return "".join(output)


def validate_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise SQLSafetyError("SQL is empty")
    if "\x00" in sql:
        raise SQLSafetyError("SQL contains a null byte")
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise SQLSafetyError("SQL comments are not allowed")

    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()
    if ";" in statement:
        raise SQLSafetyError("Multiple SQL statements are not allowed")

    masked = _mask_single_quoted_literals(statement)
    first_token_match = re.match(r"\s*([A-Za-z]+)", masked)
    first_token = first_token_match.group(1).upper() if first_token_match else ""
    if first_token not in {"SELECT", "WITH"}:
        raise SQLSafetyError("Only SELECT or WITH ... SELECT queries are allowed")
    if not re.search(r"\bSELECT\b", masked, flags=re.IGNORECASE):
        raise SQLSafetyError("A SELECT statement is required")

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", masked, flags=re.IGNORECASE):
            raise SQLSafetyError(f"Forbidden SQL keyword: {keyword}")
    if re.search(
        r"\b(?:load_extension|readfile|writefile)\s*\(",
        masked,
        flags=re.IGNORECASE,
    ):
        raise SQLSafetyError("File and extension functions are not allowed")
    return statement


def _authorizer(
    action_code: int,
    arg1: str | None,
    arg2: str | None,
    _database_name: str | None,
    _trigger_name: str | None,
) -> int:
    denied_actions = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_VTABLE,
    }
    if action_code in denied_actions:
        return sqlite3.SQLITE_DENY
    if action_code == sqlite3.SQLITE_READ and arg1 not in REAL_TABLE_NAMES:
        return sqlite3.SQLITE_DENY
    if action_code == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {
        "load_extension",
        "readfile",
        "writefile",
    }:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def execute_query(sql: str, max_rows: int = MAX_RESULT_ROWS) -> QueryResult:
    safe_sql = validate_sql(sql)
    limited_sql = f"SELECT * FROM ({safe_sql}) AS _agent_result LIMIT {max_rows + 1}"

    conn = _connect_read_only()
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

    def progress_handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    try:
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(progress_handler, 10_000)
        dataframe = pd.read_sql_query(limited_sql, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise SQLExecutionError(str(exc)) from exc
    finally:
        conn.close()

    truncated = len(dataframe) > max_rows
    if truncated:
        dataframe = dataframe.head(max_rows).copy()
    return QueryResult(dataframe=dataframe, truncated=truncated)


def _validate_business_and_execute(
    question: str,
    sql: str,
    business_context: str,
) -> tuple[QueryResult, BusinessValidationResult]:
    """Keep safety and business validation separate, in that execution order."""
    safe_sql = validate_sql(sql)
    business_validation = validate_business_rules(
        question,
        safe_sql,
        business_context,
    )
    if not business_validation.valid:
        raise BusinessRuleValidationError(business_validation)
    return execute_query(safe_sql), business_validation


def _result_json(query_result: QueryResult) -> str:
    sample = query_result.dataframe.head(LLM_RESULT_ROWS)
    rows_json = sample.to_json(orient="records", force_ascii=False)
    return json.dumps(
        {
            "returned_row_count": len(query_result.dataframe),
            "query_was_truncated_at_200_rows": query_result.truncated,
            "rows_in_this_prompt": len(sample),
            "rows": json.loads(rows_json),
        },
        ensure_ascii=False,
    )


def summarize_result(
    client: OpenAI,
    model: str,
    question: str,
    sql: str,
    query_result: QueryResult,
    business_context: str,
) -> str:
    if query_result.dataframe.empty:
        return "查询成功，但在指定条件下没有返回数据。"

    system_prompt = f"""You are a concise Chinese data analyst.
Use only the actual SQL result supplied by the user. Never invent missing values,
rows, trends, thresholds, or causal explanations. Mention important numbers and any
NULL values when relevant. If the result was truncated, explicitly say the conclusion
only covers the displayed rows.

For low-margin SKU expansion or margin-drag analysis, use cautious wording such as
“可能”, “数据显示”, “存在拖累迹象”, or “与整体毛利率下降存在一致性”. Never claim that
correlation proves causation and never invent a low-margin threshold.

Relevant business knowledge:
{business_context}
"""
    user_prompt = f"""Original question:
{question}

Executed SQL:
{sql}

Actual query result:
{_result_json(query_result)}

Provide a concise Chinese conclusion based only on this result."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty analysis conclusion")
    return content.strip()


def run_agent(
    question: str,
    trace: dict[str, Any] | None = None,
    conversation_context: dict[str, Any] | None = None,
) -> AgentResult:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    context_resolution = resolve_conversation_context(question, conversation_context)
    if context_resolution.clarification_question:
        plan = SQLPlan(
            sql=None,
            reasoning_summary="当前指代缺少唯一、可靠的上一轮实体。",
            chart_type="none",
            status="needs_clarification",
            clarification_question=context_resolution.clarification_question,
            ambiguity_type="context_reference",
        )
        if trace is not None:
            trace["first_plan"] = plan
            trace["final_plan"] = plan
            trace["repair_triggered"] = False
            trace["clarification_question"] = plan.clarification_question
            trace["ambiguity_type"] = plan.ambiguity_type
            trace["conversation_context_used"] = False
        return AgentResult(
            question=question,
            first_plan=plan,
            plan=plan,
            query_result=None,
            conclusion=None,
            repair_triggered=False,
            first_error_type=None,
            first_error_message=None,
            business_validation=None,
            turn_context=None,
            conversation_context_used=False,
        )

    config = load_config()
    client = create_llm_client(config)
    business_context = load_business_context()
    schema_context = load_schema_context()

    plan = generate_sql_plan(
        client,
        config.model,
        question,
        schema_context,
        business_context,
        context_resolution.prompt_context,
    )
    first_plan = plan
    repair_triggered = False
    first_error_type = None
    first_error_message = None
    if trace is not None:
        trace["first_plan"] = first_plan
        trace["repair_triggered"] = False
        trace["conversation_context_used"] = context_resolution.use_context

    if plan.status == "needs_clarification":
        if trace is not None:
            trace["final_plan"] = plan
            trace["clarification_question"] = plan.clarification_question
            trace["ambiguity_type"] = plan.ambiguity_type
        return AgentResult(
            question=question,
            first_plan=first_plan,
            plan=plan,
            query_result=None,
            conclusion=None,
            repair_triggered=False,
            first_error_type=None,
            first_error_message=None,
            business_validation=None,
            turn_context=None,
            conversation_context_used=context_resolution.use_context,
        )

    if plan.sql is None:
        raise LLMResponseError("A ready plan is missing SQL")

    try:
        query_result, business_validation = _validate_business_and_execute(
            question,
            plan.sql,
            business_context,
        )
    except (
        SQLSafetyError,
        BusinessRuleValidationError,
        SQLExecutionError,
    ) as first_error:
        repair_triggered = True
        first_error_type = type(first_error).__name__
        first_error_message = str(first_error)
        if trace is not None:
            trace["repair_triggered"] = True
            trace["first_error_type"] = first_error_type
            trace["first_error_message"] = first_error_message
            if isinstance(first_error, BusinessRuleValidationError):
                trace["first_business_validation"] = first_error.result
        plan = repair_sql_plan(
            client,
            config.model,
            question,
            plan,
            str(first_error),
            schema_context,
            business_context,
            context_resolution.prompt_context,
        )
        if plan.status != "ready" or plan.sql is None:
            raise LLMResponseError("SQL repair unexpectedly requested clarification")
        query_result, business_validation = _validate_business_and_execute(
            question,
            plan.sql,
            business_context,
        )
    if trace is not None:
        trace["final_plan"] = plan
        trace["query_result"] = query_result
        trace["business_validation"] = business_validation

    conclusion = summarize_result(
        client,
        config.model,
        question,
        plan.sql,
        query_result,
        business_context,
    )
    if trace is not None:
        trace["conclusion"] = conclusion
    turn_context = extract_turn_context(
        question,
        plan.sql,
        query_result.dataframe,
    )
    if trace is not None:
        trace["turn_context"] = turn_context
    return AgentResult(
        question=question,
        first_plan=first_plan,
        plan=plan,
        query_result=query_result,
        conclusion=conclusion,
        repair_triggered=repair_triggered,
        first_error_type=first_error_type,
        first_error_message=first_error_message,
        business_validation=business_validation,
        turn_context=turn_context,
        conversation_context_used=context_resolution.use_context,
    )


def _label_column(dataframe: pd.DataFrame) -> str | None:
    preferred = (
        "store_name",
        "product_name",
        "refund_reason",
        "category",
        "order_date",
        "refund_date",
        "store_id",
        "product_id",
    )
    for column in preferred:
        if column in dataframe.columns:
            return column
    non_numeric = dataframe.select_dtypes(exclude="number").columns.tolist()
    return non_numeric[0] if non_numeric else None


def _metric_columns(dataframe: pd.DataFrame) -> list[str]:
    numeric = dataframe.select_dtypes(include="number").columns.tolist()
    comparison_pairs = (
        ("q1_sales_amount", "q2_sales_amount"),
        ("q1_sales", "q2_sales"),
        ("q1_quantity", "q2_quantity"),
        ("q1_gross_margin_rate", "q2_gross_margin_rate"),
    )
    for pair in comparison_pairs:
        if all(column in numeric for column in pair):
            return list(pair)

    preferred = (
        "sales_amount",
        "sales_amount_sum",
        "refund_rate",
        "refund_amount",
        "refund_amount_sum",
        "growth_rate",
        "sales_growth",
        "gross_margin_rate_change",
        "q2_sales_share",
        "sales_quantity",
        "refund_count",
    )
    for column in preferred:
        if column in numeric:
            return [column]
    return numeric[:1]


def create_chart(
    dataframe: pd.DataFrame,
    chart_type: str,
    question: str,
) -> Path | None:
    if chart_type == "none" or dataframe.empty:
        return None
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    label_column = _label_column(dataframe)
    metric_columns = _metric_columns(dataframe)
    if label_column is None or not metric_columns:
        return None
    if chart_type == "pie" and (len(dataframe) > 12 or len(metric_columns) != 1):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plot_frame = dataframe.head(30).copy()
    labels = plot_frame[label_column].astype(str)
    if labels.duplicated().any() and "store_name" in plot_frame.columns:
        labels = plot_frame["store_name"].astype(str) + " / " + labels

    figure, axis = plt.subplots(figsize=(10, 6))
    if chart_type == "bar":
        plot_frame = plot_frame.assign(_label=labels).set_index("_label")
        plot_frame[metric_columns].plot(kind="bar", ax=axis)
        axis.set_xlabel(label_column)
        axis.set_ylabel(" / ".join(metric_columns))
    elif chart_type == "line":
        for metric in metric_columns:
            axis.plot(labels, plot_frame[metric], marker="o", label=metric)
        if len(metric_columns) > 1:
            axis.legend()
        axis.set_xlabel(label_column)
        axis.set_ylabel(" / ".join(metric_columns))
        axis.tick_params(axis="x", rotation=45)
    else:
        metric = metric_columns[0]
        if (plot_frame[metric].fillna(0) < 0).any() or plot_frame[metric].fillna(0).sum() <= 0:
            plt.close(figure)
            return None
        axis.pie(
            plot_frame[metric].fillna(0),
            labels=labels,
            autopct="%1.1f%%",
        )
        axis.set_ylabel("")

    axis.set_title(question[:60])
    figure.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"chart_{uuid.uuid4().hex[:8]}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
