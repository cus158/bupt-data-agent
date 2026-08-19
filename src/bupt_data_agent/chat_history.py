"""Small local SQLite store for persistent Streamlit conversation history."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any

from .paths import CHAT_HISTORY_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _connect() -> sqlite3.Connection:
    CHAT_HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHAT_HISTORY_DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_chat_history() -> None:
    with closing(_connect()) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_turn_context_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, message_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);
            """
        )
        _refresh_conversation_titles(connection)


def _limit_title(title: str, max_length: int = 18) -> str:
    compact = " ".join(title.split()).strip(" ，。！？；：,.!?;:") or "业务数据查询"
    return compact if len(compact) <= max_length else compact[: max_length - 1] + "…"


def _question_time_label(question: str) -> str:
    year_match = re.search(r"(?:19|20)\d{2}", question)
    year = year_match.group(0) if year_match else ""
    if re.search(r"第一季度.*第二季度|第二季度.*第一季度|Q1.*Q2|Q2.*Q1", question, re.IGNORECASE):
        return "Q1/Q2"
    if "第二季度" in question or re.search(r"\bQ2\b", question, re.IGNORECASE):
        return f"{year}Q2" if year else "Q2"
    if "第一季度" in question or re.search(r"\bQ1\b", question, re.IGNORECASE):
        return f"{year}Q1" if year else "Q1"
    if "上半年" in question or re.search(r"\bH1\b", question, re.IGNORECASE):
        return f"{year}H1" if year else "H1"
    return year


def conversation_title(question: str, max_length: int = 18) -> str:
    """Create a deterministic topic title without copying the raw question."""
    compact = " ".join(question.split())
    upper = compact.upper()
    time_label = _question_time_label(compact)

    if "SKU" in upper and "华东" in compact and ("O2O" in upper or "即时零售" in compact):
        if re.search(r"TOP\s*3|最好的\s*3|3\s*个", upper):
            return "华东 O2O Top3 SKU"
    if "毛利率下降" in compact and "增长" in compact:
        return "Q2增长与毛利下钻"
    if "退损" in compact:
        return "高退损门店分析" if "超过" in compact or "较高" in compact else "门店退损分析"
    if time_label == "Q1/Q2" and "门店" in compact and any(
        token in compact for token in ("销售额", "成交销额", "销额")
    ):
        return "Q1/Q2 门店销售对比"
    if "门店" in compact and any(
        token in compact for token in ("销售额", "成交销额", "销额")
    ):
        task = "排名" if any(token in compact for token in ("排序", "最高", "从高到低")) else "分析"
        prefix = f"{time_label} " if time_label else ""
        return _limit_title(f"{prefix}门店销额{task}", max_length)

    object_label = next(
        (label for token, label in (("SKU", "SKU"), ("门店", "门店"), ("退款", "退款"), ("商品", "商品")) if token in upper),
        "业务数据",
    )
    metric_label = next(
        (label for token, label in (("毛利率", "毛利"), ("退损率", "退损"), ("成交销额", "销额"), ("销售额", "销额"), ("销量", "销量")) if token in compact),
        "",
    )
    task_label = next(
        (label for token, label in (("增长", "增长分析"), ("比较", "对比"), ("趋势", "趋势"), ("最高", "排名"), ("排序", "排名")) if token in compact),
        "查询",
    )
    prefix = f"{time_label} " if time_label else ""
    return _limit_title(f"{prefix}{object_label}{metric_label}{task_label}", max_length)


def semantic_conversation_title(semantic_plan: dict[str, Any] | None) -> str | None:
    """Build a short deterministic title from an existing Semantic Plan."""
    if not isinstance(semantic_plan, dict):
        return None
    intent = str(semantic_plan.get("intent") or "").lower()
    metrics = [str(item) for item in semantic_plan.get("metrics") or []]
    dimensions = [str(item) for item in semantic_plan.get("dimensions") or []]
    filters = [str(item) for item in semantic_plan.get("filters") or []]
    time_range = str(semantic_plan.get("time_range") or "")
    combined = " ".join([intent, time_range, *metrics, *dimensions, *filters])
    upper_combined = combined.upper()
    compares_q1_q2 = bool(
        re.search(r"Q1.*Q2|Q2.*Q1", upper_combined)
        or ("第一季度" in combined and "第二季度" in combined)
    )

    if (
        "SKU" in upper_combined
        and "华东" in combined
        and "O2O" in upper_combined
        and ("TOP3" in upper_combined.replace(" ", "") or "TOP 3" in upper_combined)
    ):
        return "华东 O2O Top3 SKU"
    if "退损" in combined and "门店" in combined:
        return "高退损门店分析"
    if "毛利率" in combined and "增长" in combined and "SKU" in upper_combined:
        return "Q2增长与毛利下钻"
    if "门店" in combined and compares_q1_q2 and (
        "comparison" in intent or "增长" in combined or "对比" in combined
    ):
        return "Q1/Q2 门店销售对比"
    if "门店" in combined and any(metric in combined for metric in ("成交销额", "销售额", "销额")):
        year_match = re.search(r"(?:19|20)\d{2}", time_range)
        is_h1 = not compares_q1_q2 and bool(
            re.search(r"H1|上半年", time_range, re.IGNORECASE)
            or ("01-01" in time_range and ("06-30" in time_range or "07-01" in time_range))
        )
        prefix = f"{year_match.group(0)}H1 " if year_match and is_h1 else ""
        suffix = "排名" if "ranking" in intent or "排名" in combined else "分析"
        return _limit_title(f"{prefix}门店销额{suffix}")

    parts: list[str] = []
    if time_range:
        parts.append(time_range)
    if dimensions:
        parts.append(dimensions[0])
    if metrics:
        parts.append(metrics[0])
    if intent:
        parts.append("分析")
    title = " ".join(parts)
    return _limit_title(title) if title else None


def _payload_semantic_plan(payload_json: str | None) -> dict[str, Any] | None:
    payload = _decode_json(payload_json)
    if not isinstance(payload, dict):
        return None
    direct_plan = payload.get("semantic_plan")
    if isinstance(direct_plan, dict):
        return direct_plan
    details = payload.get("details")
    if isinstance(details, dict) and isinstance(details.get("semantic_plan"), dict):
        return details["semantic_plan"]
    return None


def _refresh_conversation_titles(connection: sqlite3.Connection) -> None:
    """Backfill concise titles from persisted plans or first user questions."""
    rows = connection.execute(
        """
        SELECT c.conversation_id,
               c.title,
               (
                   SELECT m.content
                   FROM messages AS m
                   WHERE m.conversation_id = c.conversation_id
                     AND m.role = 'user'
                   ORDER BY m.message_id
                   LIMIT 1
               ) AS first_question
        FROM conversations AS c
        """
    ).fetchall()
    for row in rows:
        payload_rows = connection.execute(
            """
            SELECT payload_json
            FROM messages
            WHERE conversation_id = ?
              AND role = 'assistant'
              AND payload_json IS NOT NULL
            ORDER BY message_id
            """,
            (row["conversation_id"],),
        ).fetchall()
        semantic_plan = None
        for payload_row in payload_rows:
            payload = _decode_json(payload_row["payload_json"])
            if not isinstance(payload, dict) or payload.get("kind") != "result":
                continue
            semantic_plan = _payload_semantic_plan(payload_row["payload_json"])
            if semantic_plan:
                break
        title = semantic_conversation_title(semantic_plan)
        if not title and row["first_question"]:
            title = conversation_title(row["first_question"])
        if title and title != row["title"]:
            connection.execute(
                "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                (title, row["conversation_id"]),
            )


def create_conversation(title: str) -> str:
    initialize_chat_history()
    conversation_id = uuid.uuid4().hex
    timestamp = _utc_now()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id, title, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title.strip() or "新对话", timestamp, timestamp),
        )
    return conversation_id


def update_conversation_title(conversation_id: str, title: str) -> None:
    with closing(_connect()) as connection, connection:
        connection.execute(
            "UPDATE conversations SET title = ? WHERE conversation_id = ?",
            (title.strip() or "新对话", conversation_id),
        )


def _decode_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _encode_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def append_message(
    conversation_id: str,
    role: str,
    content: str,
    payload: dict[str, Any] | None = None,
) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be 'user' or 'assistant'")
    timestamp = _utc_now()
    with closing(_connect()) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, timestamp, _encode_json(payload)),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (timestamp, conversation_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Message was not saved")
        return int(cursor.lastrowid)


def update_conversation_context(
    conversation_id: str,
    context: dict[str, Any] | None,
) -> None:
    timestamp = _utc_now()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            UPDATE conversations
            SET last_turn_context_json = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (_encode_json(context), timestamp, conversation_id),
        )


def list_conversations() -> list[dict[str, Any]]:
    initialize_chat_history()
    with closing(_connect()) as connection, connection:
        rows = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, conversation_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def load_conversation(conversation_id: str) -> dict[str, Any] | None:
    initialize_chat_history()
    with closing(_connect()) as connection, connection:
        conversation = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at,
                   last_turn_context_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        messages = connection.execute(
            """
            SELECT message_id, role, content, created_at, payload_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY message_id
            """,
            (conversation_id,),
        ).fetchall()

    return {
        "conversation_id": conversation["conversation_id"],
        "title": conversation["title"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "last_turn_context": _decode_json(conversation["last_turn_context_json"]),
        "messages": [
            {
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "payload": _decode_json(row["payload_json"]),
            }
            for row in messages
        ],
    }


def delete_conversation(conversation_id: str) -> None:
    with closing(_connect()) as connection, connection:
        connection.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
