"""Lightweight rerank service for hybrid RAG retrieval.

This is intentionally dependency-free. It behaves like a production rerank
stage in the pipeline shape: collect more candidates first, score query-doc
relevance second, then only inject the strongest evidence into the prompt.
"""

from __future__ import annotations

import re
import math
from typing import Protocol

from loguru import logger

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service


class RerankableItem(Protocol):
    kind: str
    query: str
    content: str
    source: str
    rank: int
    metadata: dict


class RerankService:
    """Score candidates with lexical relevance, channel priority and diversity."""

    DOMAIN_SOURCE_HINTS = (
        (("cpu", "处理器", "负载", "load"), ("cpu_high_usage",)),
        (("内存", "memory", "oom", "gc", "dump"), ("memory_high_usage",)),
        (("磁盘", "disk", "inode"), ("disk_high_usage",)),
        (("响应慢", "响应变慢", "变慢", "超时", "耗时", "流量突增", "slow"), ("slow_response",)),
        (("不可用", "宕机", "恢复", "打不开", "挂了", "回滚", "下游"), ("service_unavailable",)),
        (("redis", "rdb", "aof"), ("redis",)),
    )

    KIND_WEIGHT = {
        "intent_vector": 1.25,
        "keyword": 1.18,
        "supplement_vector": 0.95,
        "global_vector": 1.0,
        "tool_context": 0.85,
    }

    def strategy_name(self) -> str:
        if config.enable_semantic_rerank:
            return "hybrid_lexical_embedding"
        return "hybrid_lexical_rule"

    def rerank(self, query: str, items: list[RerankableItem], top_n: int | None = None) -> list[RerankableItem]:
        if not items:
            return []
        if not config.enable_rerank:
            return items[: (top_n or config.rerank_top_n)]

        top_n = top_n or config.rerank_top_n
        limited = items[: max(top_n, config.rerank_candidate_limit)]
        query_terms = self._terms(query)
        semantic_scores = self._semantic_scores(query, limited)

        scored: list[tuple[float, RerankableItem]] = []
        source_seen: dict[str, int] = {}
        for item in limited:
            source_key = item.source or "unknown"
            diversity_penalty = source_seen.get(source_key, 0) * 0.12
            source_seen[source_key] = source_seen.get(source_key, 0) + 1

            relevance = self._lexical_score(query_terms, item.content)
            source_relevance = self._source_score(query, item)
            semantic_relevance = semantic_scores.get(id(item), 0.0)
            channel_weight = self.KIND_WEIGHT.get(item.kind, 1.0)
            rank_bonus = 1.0 / (1 + max(0, item.rank))
            score = (
                relevance * channel_weight
                + semantic_relevance * 2.2
                + source_relevance
                + rank_bonus
                - diversity_penalty
            )
            item.metadata["rerank_score"] = round(score, 4)
            item.metadata["rerank_semantic_score"] = round(semantic_relevance, 4)
            item.metadata["rerank_lexical_score"] = round(relevance, 4)
            item.metadata["rerank_source_score"] = round(source_relevance, 4)
            item.metadata["rerank_terms"] = query_terms[:12]
            item.metadata["rerank_strategy"] = self.strategy_name()
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        selected: list[RerankableItem] = []
        tool_items: list[RerankableItem] = []
        for _, item in scored:
            if item.kind == "tool_context":
                tool_items.append(item)
                continue
            if len(selected) < top_n:
                selected.append(item)

        # Tool context is real-time evidence; keep one when present, but do not
        # let it crowd out all document evidence.
        if tool_items:
            selected.append(tool_items[0])
        return selected

    def _lexical_score(self, query_terms: list[str], content: str) -> float:
        if not query_terms or not content:
            return 0.0
        lowered = content.lower()
        content_terms = set(self._terms(content))
        score = 0.0
        for term in query_terms:
            if term in content_terms:
                score += 1.0
            if len(term) >= 2 and term in lowered:
                score += 0.6
        return score / max(1, len(query_terms))

    def _source_score(self, query: str, item: RerankableItem) -> float:
        query_lower = query.lower()
        metadata = item.metadata or {}
        source_text = " ".join(
            str(value)
            for value in (
                item.source,
                metadata.get("_file_name", ""),
                metadata.get("_source", ""),
                metadata.get("source", ""),
            )
            if value
        ).lower()

        score = 0.0
        for query_hints, source_hints in self.DOMAIN_SOURCE_HINTS:
            if any(hint.lower() in query_lower for hint in query_hints):
                if any(hint.lower() in source_text for hint in source_hints):
                    score += 1.8
                elif item.kind == "keyword":
                    score -= 0.35
        return score

    def _semantic_scores(self, query: str, items: list[RerankableItem]) -> dict[int, float]:
        if not config.enable_semantic_rerank or not query or not items:
            return {}
        try:
            texts = [
                self._prepare_semantic_text(item)
                for item in items
            ]
            query_embedding = vector_embedding_service.embed_query(query)
            doc_embeddings = vector_embedding_service.embed_documents(texts)
            return {
                id(item): max(0.0, self._cosine(query_embedding, embedding))
                for item, embedding in zip(items, doc_embeddings)
            }
        except Exception as exc:
            logger.warning("Semantic rerank failed, fallback to lexical rerank: {}", exc)
            return {}

    def _prepare_semantic_text(self, item: RerankableItem) -> str:
        metadata = item.metadata or {}
        headers = " ".join(
            str(metadata.get(key, ""))
            for key in ("_file_name", "h1", "h2", "h3")
            if metadata.get(key)
        )
        content = " ".join(str(item.content or "").split())
        limit = max(120, config.semantic_rerank_content_chars)
        return f"{item.source} {headers}\n{content[:limit]}".strip()

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _terms(self, text: str) -> list[str]:
        lowered = text.lower()
        ascii_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}|[0-9]{2,}", lowered)
        chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
        chinese_terms: list[str] = []
        for phrase in chinese_phrases:
            if len(phrase) <= 4:
                chinese_terms.append(phrase)
            for idx in range(len(phrase) - 1):
                chinese_terms.append(phrase[idx : idx + 2])
        seen: set[str] = set()
        terms: list[str] = []
        for term in ascii_terms + chinese_terms:
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
        return terms


rerank_service = RerankService()
