"""Domain term mapping for query normalization.

The goal is to make colloquial operations questions stable before intent
classification and retrieval. The rules are JSON-backed so the demo can show
an enterprise-style admin surface without requiring a database migration.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config


@dataclass
class TermMapping:
    id: str
    source_term: str
    target_term: str
    scenario: str = "AIOps"
    enabled: bool = True
    priority: int = 50
    match_type: str = "contains"
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TermMappingService:
    """Manage and apply query term mappings."""

    def __init__(self) -> None:
        self.path = Path(config.query_term_mapping_path)
        self._lock = threading.RLock()
        self._mappings: list[TermMapping] | None = None

    def list_mappings(self) -> dict[str, Any]:
        mappings = self._load()
        return {
            "path": str(self.path),
            "count": len(mappings),
            "enabled_count": sum(1 for item in mappings if item.enabled),
            "mappings": [item.to_dict() for item in mappings],
        }

    def normalize(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        if not config.enable_query_term_mapping or not text:
            return text, []

        result = text
        applied: list[dict[str, Any]] = []
        for mapping in self._sorted_enabled_mappings():
            next_result = self._apply_mapping(result, mapping)
            if next_result == result:
                continue
            applied.append(
                {
                    "id": mapping.id,
                    "source_term": mapping.source_term,
                    "target_term": mapping.target_term,
                    "scenario": mapping.scenario,
                }
            )
            result = next_result

        if applied:
            logger.info("Query term mapping applied: original={}, normalized={}, mappings={}", text, result, applied)
        return result, applied

    def upsert_mapping(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = str(payload.get("source_term") or "").strip()
        target = str(payload.get("target_term") or "").strip()
        if not source or not target:
            raise ValueError("source_term and target_term are required")

        now = datetime.now().isoformat(timespec="seconds")
        mapping_id = str(payload.get("id") or self._make_id(source, target)).strip()
        with self._lock:
            mappings = self._load()
            existing = next((item for item in mappings if item.id == mapping_id), None)
            if existing is None:
                existing = TermMapping(
                    id=mapping_id,
                    source_term=source,
                    target_term=target,
                    created_at=now,
                    updated_at=now,
                )
                mappings.append(existing)
            existing.source_term = source
            existing.target_term = target
            existing.scenario = str(payload.get("scenario") or existing.scenario or "AIOps").strip()
            existing.enabled = bool(payload.get("enabled", existing.enabled))
            existing.priority = int(payload.get("priority", existing.priority or 50))
            existing.match_type = str(payload.get("match_type") or existing.match_type or "contains").strip()
            existing.description = str(payload.get("description") or existing.description or "").strip()
            existing.updated_at = now
            self._save(mappings)
            return existing.to_dict()

    def delete_mapping(self, mapping_id: str) -> bool:
        with self._lock:
            mappings = self._load()
            kept = [item for item in mappings if item.id != mapping_id]
            if len(kept) == len(mappings):
                return False
            self._save(kept)
            return True

    def _sorted_enabled_mappings(self) -> list[TermMapping]:
        return sorted(
            [item for item in self._load() if item.enabled],
            key=lambda item: (item.priority, len(item.source_term)),
            reverse=True,
        )

    def _apply_mapping(self, text: str, mapping: TermMapping) -> str:
        source = mapping.source_term.strip()
        target = mapping.target_term.strip()
        if not source or not target:
            return text

        if mapping.match_type == "regex":
            try:
                return re.sub(source, target, text, flags=re.IGNORECASE)
            except re.error:
                logger.warning("Invalid term mapping regex skipped: id={}, pattern={}", mapping.id, source)
                return text

        pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
        return pattern.sub(target, text)

    def _load(self) -> list[TermMapping]:
        with self._lock:
            if self._mappings is not None:
                return self._mappings
            if not self.path.exists():
                self._mappings = self._default_mappings()
                self._save(self._mappings)
                return self._mappings
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                raw_items = payload.get("mappings", []) if isinstance(payload, dict) else payload
                self._mappings = [self._from_dict(item) for item in raw_items if isinstance(item, dict)]
                if not self._mappings:
                    self._mappings = self._default_mappings()
                    self._save(self._mappings)
                return self._mappings
            except Exception as exc:
                logger.warning("Failed to load term mappings {}, using defaults: {}", self.path, exc)
                self._mappings = self._default_mappings()
                return self._mappings

    def _save(self, mappings: list[TermMapping]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "query-term-mapping-v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "mappings": [item.to_dict() for item in mappings],
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
        self._mappings = mappings

    def _from_dict(self, payload: dict[str, Any]) -> TermMapping:
        now = datetime.now().isoformat(timespec="seconds")
        return TermMapping(
            id=str(payload.get("id") or self._make_id(payload.get("source_term", ""), payload.get("target_term", ""))),
            source_term=str(payload.get("source_term") or ""),
            target_term=str(payload.get("target_term") or ""),
            scenario=str(payload.get("scenario") or "AIOps"),
            enabled=bool(payload.get("enabled", True)),
            priority=int(payload.get("priority", 50)),
            match_type=str(payload.get("match_type") or "contains"),
            description=str(payload.get("description") or ""),
            created_at=str(payload.get("created_at") or now),
            updated_at=str(payload.get("updated_at") or now),
        )

    def _default_mappings(self) -> list[TermMapping]:
        now = datetime.now().isoformat(timespec="seconds")
        defaults = [
            ("cpu_high", "CPU飙高", "CPU 使用率过高", "AIOps", 100, "把口语化 CPU 告警统一到知识库标准说法"),
            ("cpu_full", "CPU打满", "CPU 使用率过高", "AIOps", 98, "常见线上告警口语"),
            ("java_cpu", "Java卡死", "Java 服务 CPU 高，线程阻塞或热点线程", "Java", 96, "引导 Java 专用排查链路"),
            ("service_slow", "服务卡", "服务响应慢", "AIOps", 90, "响应慢类问题归一化"),
            ("api_down", "接口挂了", "服务不可用", "AIOps", 90, "可用性故障归一化"),
            ("disk_full", "磁盘爆了", "磁盘空间不足", "AIOps", 88, "磁盘告警归一化"),
            ("memory_leak", "内存涨不停", "内存泄漏", "AIOps", 86, "内存问题归一化"),
            ("jvm_gc", "频繁 Full GC", "JVM GC 频繁", "Java", 84, "Java 内存/GC 场景归一化"),
        ]
        return [
            TermMapping(
                id=mapping_id,
                source_term=source,
                target_term=target,
                scenario=scenario,
                priority=priority,
                description=description,
                created_at=now,
                updated_at=now,
            )
            for mapping_id, source, target, scenario, priority, description in defaults
        ]

    @staticmethod
    def _make_id(source: Any, target: Any) -> str:
        raw = f"{source}_{target}".strip().lower()
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", raw).strip("_")
        return normalized[:80] or f"mapping_{datetime.now().strftime('%Y%m%d%H%M%S')}"


term_mapping_service = TermMappingService()
