"""Persistent long-term user memory.

Short-term memory is session-scoped and optimized for the current conversation.
This service stores stable user preferences and recurring topics across sessions
so a new chat can still inherit the user's learning goals and preferred style.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config


class LongTermMemoryService:
    """JSON-backed long-term user memory for the local demo."""

    def __init__(self) -> None:
        self.path = Path(config.long_term_memory_path)
        self._lock = threading.RLock()
        self._store: dict[str, Any] | None = None

    def normalize_user_id(self, user_id: str | None = None) -> str:
        raw = (user_id or config.auth_username or "default").strip() or "default"
        normalized = re.sub(r"[^a-zA-Z0-9_.@-]+", "_", raw)
        return normalized[:80] or "default"

    def get_user_memory(self, user_id: str | None = None) -> dict[str, Any]:
        if not config.enable_long_term_user_memory:
            return self._empty_user_memory(self.normalize_user_id(user_id))
        with self._lock:
            store = self._load_store()
            normalized_user_id = self.normalize_user_id(user_id)
            users = store.setdefault("users", {})
            if normalized_user_id not in users:
                users[normalized_user_id] = self._empty_user_memory(normalized_user_id)
            return dict(users[normalized_user_id])

    def format_context(
        self,
        user_id: str | None = None,
        *,
        intent: str = "",
        allow_topic_recall: bool = False,
    ) -> str:
        """Format long-term memory for prompt injection.

        Long-term memory is split into two layers:
        1. safe preferences, such as beginner-friendly explanation style;
        2. topical recall, such as previous RAG/eval sessions.

        Topical recall must be explicitly allowed by the query-understanding
        layer; otherwise old project topics can pollute a fresh AIOps question.
        """
        if not config.enable_long_term_user_memory:
            return ""
        memory = self.get_user_memory(user_id)
        parts: list[str] = []
        include_project_preferences = intent == "knowledge_query" or allow_topic_recall
        if allow_topic_recall and memory.get("summary"):
            parts.append(f"长期用户摘要: {memory['summary']}")
        if memory.get("user_profile"):
            profile_items = self._format_profile_items(
                memory["user_profile"],
                include_project_preferences=include_project_preferences,
            )
            if profile_items:
                parts.append("长期用户画像: " + "；".join(profile_items[:8]))
        if memory.get("stable_facts"):
            stable_facts = self._filter_stable_facts(
                memory["stable_facts"],
                include_project_preferences=include_project_preferences,
            )
            if stable_facts:
                parts.append("长期稳定信息: " + "；".join(stable_facts[-8:]))
        if allow_topic_recall and memory.get("topic_stack"):
            parts.append("长期主题轨迹: " + " -> ".join(memory["topic_stack"][-6:]))
        if allow_topic_recall and memory.get("entities"):
            entity_lines = []
            for entity_type, values in memory["entities"].items():
                if values:
                    entity_lines.append(f"{entity_type}: {', '.join(values[-8:])}")
            if entity_lines:
                parts.append("长期关键实体: " + "；".join(entity_lines[:6]))
        if allow_topic_recall and memory.get("focus_questions"):
            parts.append("长期关注问题: " + "；".join(memory["focus_questions"][-5:]))
        if allow_topic_recall and memory.get("recent_sessions"):
            session_lines = []
            for item in memory["recent_sessions"][-3:]:
                session_lines.append(
                    f"- session={item.get('session_id', '')}, topic={item.get('last_topic', '')}, "
                    f"summary={item.get('summary', '')}"
                )
            if session_lines:
                parts.append("最近会话摘要:\n" + "\n".join(session_lines))
        return "\n".join(parts)

    def _format_profile_items(
        self,
        profile: dict[str, Any],
        *,
        include_project_preferences: bool,
    ) -> list[str]:
        safe_keys = {"explanation_level", "answer_style"}
        profile_items = []
        for key, value in profile.items():
            if not include_project_preferences and key not in safe_keys:
                continue
            if isinstance(value, list):
                profile_items.append(f"{key}={','.join(str(item) for item in value[-5:])}")
            else:
                profile_items.append(f"{key}={value}")
        return profile_items

    def _filter_stable_facts(
        self,
        facts: list[Any],
        *,
        include_project_preferences: bool,
    ) -> list[str]:
        if include_project_preferences:
            return [str(item) for item in facts if str(item).strip()]

        project_markers = (
            "项目",
            "面试",
            "简历",
            "量化",
            "评测",
            "Baseline",
            "baseline",
            "页面",
            "界面",
            "展示",
            "同质化",
            "技术深度",
        )
        safe_facts = []
        for fact in facts:
            text = str(fact).strip()
            if not text:
                continue
            if any(marker in text for marker in project_markers):
                continue
            safe_facts.append(text)
        return safe_facts

    def update_from_session(
        self,
        user_id: str | None,
        session_id: str,
        session_memory: dict[str, Any],
        question: str,
        answer_preview: str,
    ) -> dict[str, Any]:
        """Merge one session memory snapshot into long-term user memory."""
        if not config.enable_long_term_user_memory:
            return self._empty_user_memory(self.normalize_user_id(user_id))

        with self._lock:
            store = self._load_store()
            normalized_user_id = self.normalize_user_id(user_id)
            users = store.setdefault("users", {})
            memory = users.setdefault(normalized_user_id, self._empty_user_memory(normalized_user_id))
            now = datetime.now().isoformat(timespec="seconds")

            memory["updated_at"] = now
            memory["turn_count"] = int(memory.get("turn_count", 0)) + 1
            memory["stable_facts"] = self._merge_list(
                memory.get("stable_facts", []),
                session_memory.get("stable_facts", []),
                limit=24,
            )
            memory["topic_stack"] = self._merge_list(
                memory.get("topic_stack", []),
                session_memory.get("topic_stack", []),
                limit=18,
            )
            memory["focus_questions"] = self._merge_list(
                memory.get("focus_questions", []),
                session_memory.get("focus_questions", []),
                limit=18,
            )
            memory["user_profile"] = self._merge_profile(
                memory.get("user_profile", {}),
                session_memory.get("user_profile", {}),
            )
            memory["entities"] = self._merge_entities(
                memory.get("entities", {}),
                session_memory.get("entities", {}),
            )

            recent_sessions = [
                item
                for item in memory.get("recent_sessions", [])
                if item.get("session_id") != session_id
            ]
            recent_sessions.append(
                {
                    "session_id": session_id,
                    "last_topic": session_memory.get("last_topic", ""),
                    "summary": session_memory.get("summary", ""),
                    "last_question": question,
                    "last_answer_preview": answer_preview,
                    "updated_at": now,
                }
            )
            memory["recent_sessions"] = recent_sessions[-config.long_term_memory_recent_sessions :]
            memory["summary"] = self._build_summary(memory)

            self._save_store(store)
            logger.debug(
                "Long-term memory updated: user={}, session={}, topics={}",
                normalized_user_id,
                session_id,
                len(memory.get("topic_stack", [])),
            )
            return dict(memory)

    def clear_user_memory(self, user_id: str | None = None) -> bool:
        normalized_user_id = self.normalize_user_id(user_id)
        with self._lock:
            store = self._load_store()
            users = store.setdefault("users", {})
            users.pop(normalized_user_id, None)
            self._save_store(store)
        return True

    def _load_store(self) -> dict[str, Any]:
        if self._store is not None:
            return self._store
        if not self.path.exists():
            self._store = {"version": "long-term-memory-v1", "users": {}}
            return self._store
        try:
            self._store = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self._store, dict):
                raise ValueError("memory store is not an object")
            self._store.setdefault("version", "long-term-memory-v1")
            self._store.setdefault("users", {})
            return self._store
        except Exception as exc:
            logger.warning("Failed to load long-term memory {}, using empty store: {}", self.path, exc)
            self._store = {"version": "long-term-memory-v1", "users": {}}
            return self._store

    def _save_store(self, store: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
        self._store = store

    def _empty_user_memory(self, user_id: str) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        return {
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "summary": "",
            "stable_facts": [],
            "topic_stack": [],
            "user_profile": {},
            "entities": {},
            "focus_questions": [],
            "recent_sessions": [],
            "turn_count": 0,
        }

    def _build_summary(self, memory: dict[str, Any]) -> str:
        parts: list[str] = []
        if memory.get("stable_facts"):
            parts.append("用户长期偏好/背景: " + "；".join(memory["stable_facts"][-8:]) + "。")
        if memory.get("topic_stack"):
            parts.append("长期关注主题: " + " -> ".join(memory["topic_stack"][-6:]) + "。")
        if memory.get("focus_questions"):
            parts.append("近期持续关注: " + "；".join(memory["focus_questions"][-5:]) + "。")
        if memory.get("entities"):
            entity_parts = []
            for entity_type, values in memory["entities"].items():
                if values:
                    entity_parts.append(f"{entity_type}={','.join(values[-5:])}")
            if entity_parts:
                parts.append("常见关键实体: " + "；".join(entity_parts[:6]) + "。")
        return self._compact_text("".join(parts), config.enhanced_memory_summary_max_chars)

    @staticmethod
    def _merge_list(existing: list[Any], incoming: list[Any], limit: int) -> list[Any]:
        result = list(existing or [])
        for item in incoming or []:
            normalized = str(item).strip()
            if not normalized:
                continue
            result = [old for old in result if old != normalized]
            result.append(normalized)
        return result[-limit:]

    def _merge_profile(self, existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        result = dict(existing or {})
        for key, value in (incoming or {}).items():
            if isinstance(value, list):
                current = result.get(key, [])
                if not isinstance(current, list):
                    current = [str(current)]
                result[key] = self._merge_list(current, value, limit=12)
            elif isinstance(value, dict):
                current = result.get(key, {})
                if not isinstance(current, dict):
                    current = {}
                current.update(value)
                result[key] = current
            elif value not in ("", None):
                result[key] = value
        return result

    def _merge_entities(self, existing: dict[str, list[str]], incoming: dict[str, list[str]]) -> dict[str, list[str]]:
        result = dict(existing or {})
        for entity_type, values in (incoming or {}).items():
            current = result.get(entity_type, [])
            result[entity_type] = self._merge_list(current, values, limit=max(1, config.enhanced_memory_entity_limit))
        return result

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        compacted = " ".join(str(text).strip().split())
        if len(compacted) <= max_chars:
            return compacted
        return compacted[: max_chars - 3] + "..."


long_term_memory_service = LongTermMemoryService()
