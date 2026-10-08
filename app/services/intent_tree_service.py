"""Configurable 运维 intent tree.

The query-understanding layer still has deterministic fallback rules, while
this service adds a Ragent-style intent tree: every node describes one
diagnosis / knowledge scope and can bind retrieval source patterns.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config


@dataclass
class IntentNode:
    node_id: str
    name: str
    parent: str
    level: str
    route: str
    keywords: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source_patterns: list[str] = field(default_factory=list)
    guidance_label: str = ""

    @property
    def path(self) -> str:
        return f"{self.parent} > {self.name}" if self.parent else self.name


@dataclass
class IntentNodeMatch:
    node_id: str
    name: str
    parent: str
    level: str
    route: str
    score: float
    path: str
    matched_terms: list[str] = field(default_factory=list)
    source_patterns: list[str] = field(default_factory=list)
    guidance_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntentTreeService:
    """Load and score a configurable intent tree."""

    DEFAULT_NODES: list[dict[str, Any]] = [
        {
            "id": "ops.cpu",
            "name": "CPU 高负载排查",
            "parent": "AIOps 故障诊断",
            "level": "fault",
            "route": "aiops_diagnosis",
            "keywords": ["CPU", "cpu", "负载", "load", "使用率"],
            "aliases": ["CPU过高", "机器负载高"],
            "examples": ["CPU 过高怎么排查？"],
            "source_patterns": ["cpu_high_usage", "cpu"],
            "guidance_label": "CPU 使用率过高 / 机器负载高",
        },
        {
            "id": "ops.memory",
            "name": "内存异常排查",
            "parent": "AIOps 故障诊断",
            "level": "fault",
            "route": "aiops_diagnosis",
            "keywords": ["内存", "memory", "OOM", "oom", "泄漏", "GC"],
            "aliases": ["内存过高", "内存泄漏"],
            "examples": ["内存泄漏怎么定位？"],
            "source_patterns": ["memory_high_usage", "memory"],
            "guidance_label": "内存过高 / OOM / GC",
        },
        {
            "id": "project.rag",
            "name": "RAG / Agent 项目实现",
            "parent": "项目知识问答",
            "level": "project",
            "route": "knowledge_query",
            "keywords": ["RAG", "rag", "Agent", "MCP", "Milvus", "重排"],
            "aliases": ["RAG 项目", "Agent 项目"],
            "examples": ["这个项目 RAG 是怎么做的？"],
            "source_patterns": ["RAG", "MCP", "Agent", "项目"],
            "guidance_label": "RAG / Agent / MCP 项目实现",
        },
    ]

    def __init__(self) -> None:
        self.path = Path(config.intent_tree_path)
        self._nodes: list[IntentNode] | None = None

    def resolve(self, question: str, *, limit: int | None = None) -> list[IntentNodeMatch]:
        """Return ranked intent-node candidates for one question."""
        if not question.strip():
            return []

        nodes = self._load_nodes()
        matches: list[IntentNodeMatch] = []
        for node in nodes:
            score, matched_terms = self._score(question, node)
            if score < config.intent_tree_min_score:
                continue
            matches.append(
                IntentNodeMatch(
                    node_id=node.node_id,
                    name=node.name,
                    parent=node.parent,
                    level=node.level,
                    route=node.route,
                    score=round(score, 3),
                    path=node.path,
                    matched_terms=matched_terms,
                    source_patterns=node.source_patterns,
                    guidance_label=node.guidance_label or node.name,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: max(1, limit or config.intent_tree_max_candidates)]

    def resolve_many(self, questions: list[str], *, limit: int | None = None) -> list[IntentNodeMatch]:
        """Resolve multiple sub-questions and merge candidates by node id."""
        merged: dict[str, IntentNodeMatch] = {}
        for question in questions:
            for match in self.resolve(question, limit=limit):
                existing = merged.get(match.node_id)
                if not existing or match.score > existing.score:
                    merged[match.node_id] = match
                elif existing:
                    existing.matched_terms = self._dedupe([*existing.matched_terms, *match.matched_terms])
        matches = list(merged.values())
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: max(1, limit or config.intent_tree_max_candidates)]

    def node_catalog(self) -> list[dict[str, Any]]:
        """Return the configured tree nodes for LLM candidate scoring."""
        return [
            {
                "id": node.node_id,
                "name": node.name,
                "path": node.path,
                "level": node.level,
                "route": node.route,
                "description": node.guidance_label or node.name,
                "keywords": node.keywords[:8],
                "aliases": node.aliases[:6],
                "examples": node.examples[:4],
                "source_patterns": node.source_patterns,
            }
            for node in self._load_nodes()
            if node.node_id
        ]

    def match_from_score(
        self,
        node_id: str,
        score: float,
        *,
        reason: str = "",
        matched_terms: list[str] | None = None,
    ) -> IntentNodeMatch | None:
        """Build a safe IntentNodeMatch from an LLM-scored node id."""
        node = self._node_by_id(node_id)
        if not node:
            return None
        score = max(0.0, min(1.0, float(score)))
        terms = self._dedupe([*(matched_terms or []), reason])[:8]
        return IntentNodeMatch(
            node_id=node.node_id,
            name=node.name,
            parent=node.parent,
            level=node.level,
            route=node.route,
            score=round(score, 3),
            path=node.path,
            matched_terms=terms,
            source_patterns=node.source_patterns,
            guidance_label=node.guidance_label or node.name,
        )

    def build_retrieval_scope(self, matches: list[IntentNodeMatch]) -> dict[str, Any]:
        """Build a retrieval scope from top intent-node matches."""
        if not matches:
            return {
                "mode": "global",
                "top_score": 0.0,
                "target_sources": [],
                "supplement_ratio": 0.0,
                "reason": "no_intent_tree_match",
            }

        top = matches[0]
        target_sources = self._dedupe(
            [
                pattern
                for match in matches
                if match.score >= max(config.intent_tree_min_score, top.score * 0.72)
                for pattern in match.source_patterns
            ]
        )
        mode = "directed_with_supplement" if target_sources and top.score >= 0.55 else "global"
        return {
            "mode": mode,
            "top_score": top.score,
            "top_node": top.to_dict(),
            "target_sources": target_sources,
            "supplement_ratio": config.retrieval_supplement_ratio if mode == "directed_with_supplement" else 0.0,
            "reason": "high_confidence_intent_scope" if mode == "directed_with_supplement" else "low_confidence_global_scope",
        }

    def maybe_ambiguity_prompt(self, question: str, matches: list[IntentNodeMatch]) -> str:
        """Return a clarification prompt when top candidates are close."""
        if len(matches) < 2:
            return ""

        top = matches[0]
        second = matches[1]
        if top.score <= 0:
            return ""
        if second.score / top.score < config.intent_tree_ambiguity_ratio:
            return ""
        if self._has_explicit_node_signal(question, top):
            return ""

        options = []
        for idx, match in enumerate(matches[:4], start=1):
            label = match.guidance_label or match.name
            options.append(f"{idx}. {label}")
        return "我识别到你的问题可能对应多个方向，请先确认你想问哪一个：\n" + "\n".join(options)

    def _load_nodes(self) -> list[IntentNode]:
        if self._nodes is not None:
            return self._nodes

        raw_nodes = self.DEFAULT_NODES
        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                loaded = payload.get("nodes", []) if isinstance(payload, dict) else payload
                if isinstance(loaded, list) and loaded:
                    raw_nodes = loaded
        except Exception as exc:
            logger.warning("Failed to load intent tree {}, using defaults: {}", self.path, exc)

        self._nodes = [self._node_from_payload(item) for item in raw_nodes if isinstance(item, dict)]
        return self._nodes

    def _node_from_payload(self, payload: dict[str, Any]) -> IntentNode:
        return IntentNode(
            node_id=str(payload.get("id") or payload.get("node_id") or "").strip(),
            name=str(payload.get("name") or "").strip(),
            parent=str(payload.get("parent") or "").strip(),
            level=str(payload.get("level") or "topic").strip(),
            route=str(payload.get("route") or "knowledge_query").strip(),
            keywords=self._list_of_str(payload.get("keywords")),
            aliases=self._list_of_str(payload.get("aliases")),
            examples=self._list_of_str(payload.get("examples")),
            source_patterns=self._list_of_str(payload.get("source_patterns")),
            guidance_label=str(payload.get("guidance_label") or payload.get("name") or "").strip(),
        )

    def _node_by_id(self, node_id: str) -> IntentNode | None:
        node_id = str(node_id or "").strip()
        for node in self._load_nodes():
            if node.node_id == node_id:
                return node
        return None

    def _score(self, question: str, node: IntentNode) -> tuple[float, list[str]]:
        normalized_question = self._normalize(question)
        matched: list[str] = []
        score = 0.0

        for term in node.keywords:
            if self._contains(normalized_question, term):
                matched.append(term)
                score += 0.16

        for alias in node.aliases:
            if self._contains(normalized_question, alias):
                matched.append(alias)
                score += 0.24

        for example in node.examples:
            overlap = self._token_overlap(question, example)
            if overlap >= 2:
                matched.append(f"example:{example[:18]}")
                score += min(0.22, 0.06 * overlap)

        node_name_terms = [node.name, node.guidance_label, node.parent]
        for term in node_name_terms:
            if term and self._contains(normalized_question, term):
                matched.append(term)
                score += 0.18

        if matched:
            score += 0.28
        if node.route == "aiops_diagnosis" and any(term in normalized_question for term in ("排查", "故障", "告警", "诊断")):
            score += 0.08
        if node.route == "knowledge_query" and any(term in normalized_question for term in ("什么是", "讲解", "实现", "面试")):
            score += 0.08

        return min(score, 0.99), self._dedupe(matched)[:8]

    def _has_explicit_node_signal(self, question: str, match: IntentNodeMatch) -> bool:
        text = self._normalize(question)
        for term in [match.name, match.guidance_label, *match.matched_terms]:
            if term and not term.startswith("example:") and self._contains(text, term):
                return True
        return False

    @staticmethod
    def _contains(normalized_text: str, term: str) -> bool:
        normalized_term = IntentTreeService._normalize(term)
        return bool(normalized_term and normalized_term in normalized_text)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[\s,，。！？?;；:：'\"“”‘’()\[\]{}<>《》_-]+", "", str(text).lower())

    @staticmethod
    def _token_overlap(left: str, right: str) -> int:
        pattern = r"[a-zA-Z0-9_@+-]+|[\u4e00-\u9fff]{2,}"
        left_tokens = set(re.findall(pattern, left.lower()))
        right_tokens = set(re.findall(pattern, right.lower()))
        return len(left_tokens & right_tokens)

    @staticmethod
    def _list_of_str(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result


intent_tree_service = IntentTreeService()
