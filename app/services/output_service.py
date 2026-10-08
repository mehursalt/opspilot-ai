"""Output service for lightweight message persistence and trace logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config


class OutputService:
    """Persist chat messages and RAG pipeline traces as JSONL files."""

    def __init__(self) -> None:
        self.logs_dir = Path("logs")
        self.message_file = self.logs_dir / "rag_messages.jsonl"
        self.trace_file = self.logs_dir / "rag_traces.jsonl"

    def generate_title(self, question: str) -> str:
        title = " ".join(question.strip().split())
        if not title:
            return "未命名会话"
        return title[:24] + ("..." if len(title) > 24 else "")

    def persist_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._append_jsonl(
            self.message_file,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            },
        )

    def record_trace(self, session_id: str, event: str, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.trace_file,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "session_id": session_id,
                "event": event,
                "payload": payload,
            },
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        if not config.enable_output_trace:
            return
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning("输出服务写入失败 {}: {}", path, e)


output_service = OutputService()
