"""RAG retrieval engine.

This service follows the "parallel retrieval" part of the target architecture:
intent-oriented vector search, global vector search and tool-context retrieval
are executed before generation, then deduplicated and formatted as prompt
context.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.keyword_search_service import KeywordSearchHit, keyword_search_service
from app.services.query_understanding_service import QueryUnderstandingResult
from app.services.rerank_service import rerank_service
from app.services.vector_store_manager import vector_store_manager
from app.tools.query_metrics_alerts import query_prometheus_alerts


@dataclass
class RetrievalItem:
    """Single retrieval item from vector store or tools."""

    kind: str
    query: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    score: float = 0.0


@dataclass
class RetrievalEngineResult:
    """Aggregated retrieval result used by prompt assembly."""

    items: list[RetrievalItem] = field(default_factory=list)
    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def json_safe(value: Any) -> Any:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))

        return {
            "items": [json_safe(asdict(item)) for item in self.items],
            "context": self.context,
            "sources": json_safe(self.sources),
            "trace": json_safe(self.trace),
        }


class RetrievalEngineService:
    """Pre-generation retrieval engine."""

    async def retrieve(self, understanding: QueryUnderstandingResult) -> RetrievalEngineResult:
        if not config.enable_rag_prefetch:
            return RetrievalEngineResult(trace={"enabled": False})

        channel_tasks: list[tuple[str, asyncio.Task[dict[str, Any]]]] = []
        search_queries = understanding.search_queries or [understanding.rewritten_question]
        retrieval_scope = understanding.retrieval_scope or {}
        target_sources = [
            str(item).strip()
            for item in retrieval_scope.get("target_sources", [])
            if str(item).strip()
        ]
        strategies: list[str] = []

        # Intent-directed retrieval: prioritize sources matched by the intent tree.
        strategies.append("intent_vector")
        primary_query = search_queries[0]
        channel_tasks.append(
            (
                "intent_vector",
                asyncio.create_task(
                    self._run_channel(
                        "intent_vector",
                        self._vector_search(
                            primary_query,
                            kind="intent_vector",
                            rank_offset=0,
                            source_patterns=target_sources,
                        ),
                    )
                ),
            )
        )

        # Supplement retrieval: keep a small quota for non-target sources to avoid intent misses.
        supplement_ratio = float(retrieval_scope.get("supplement_ratio") or 0.0)
        if target_sources and supplement_ratio > 0:
            strategies.append("supplement_vector")
            channel_tasks.append(
                (
                    "supplement_vector",
                    asyncio.create_task(
                        self._run_channel(
                            "supplement_vector",
                            self._vector_search(
                                primary_query,
                                kind="supplement_vector",
                                rank_offset=50,
                                exclude_source_patterns=target_sources,
                            ),
                        )
                    ),
                )
            )

        # Global vector retrieval: search the remaining candidate queries.
        global_vector_tasks = [
            asyncio.create_task(
                self._vector_search(query, kind="global_vector", rank_offset=idx * 100)
            )
            for idx, query in enumerate(search_queries[1:], start=1)
        ]
        if global_vector_tasks:
            strategies.append("global_vector")
            channel_tasks.append(
                (
                    "global_vector",
                    asyncio.create_task(
                        self._run_channel(
                            "global_vector",
                            self._merge_channel_tasks(global_vector_tasks),
                        )
                    ),
                )
            )

        # Keyword retrieval: exact match for alert names, error codes and service names.
        if config.enable_keyword_retrieval:
            strategies.append("keyword")
            channel_tasks.append(
                (
                    "keyword",
                    asyncio.create_task(self._run_channel("keyword", self._keyword_search(search_queries))),
                )
            )

        # Tool context retrieval: prefetch Prometheus alerts for ops/metrics questions.
        if understanding.intent in ("aiops_diagnosis", "metrics_alerts"):
            strategies.append("tool_context")
            channel_tasks.append(
                (
                    "tool_context",
                    asyncio.create_task(self._run_channel("tool_context", self._tool_context(understanding))),
                )
            )

        collected: list[RetrievalItem] = []
        errors: list[str] = []
        channel_runs: list[dict[str, Any]] = []
        for task in asyncio.as_completed([item[1] for item in channel_tasks]):
            try:
                run = await task
                channel_runs.append(run)
                collected.extend(run.get("items", []))
                if run.get("error"):
                    errors.append(str(run["error"]))
            except Exception as e:
                errors.append(str(e))
                logger.warning("检索子任务失败: {}", e)

        deduped = self._dedupe_items(collected)
        fused = self._fuse_by_rrf(deduped, channel_runs)
        items = rerank_service.rerank(
            understanding.rewritten_question,
            fused,
            top_n=config.rerank_top_n,
        )
        self._annotate_final_ranks(items)
        sources = self._build_sources(items)
        context = self._format_context(items)
        channel_summary = self._build_channel_summary(channel_runs, items)
        trace = {
            "enabled": True,
            "query_count": len(search_queries),
            "raw_item_count": len(collected),
            "deduped_item_count": len(deduped),
            "fused_candidate_count": len(fused),
            "final_item_count": len(items),
            "term_mapping_applied": understanding.applied_mappings,
            "rerank_enabled": config.enable_rerank,
            "rerank_strategy": rerank_service.strategy_name(),
            "rrf_enabled": config.enable_rrf_fusion,
            "rrf_k": config.rrf_k,
            "semantic_rerank_enabled": config.enable_semantic_rerank,
            "retrieval_scope": retrieval_scope,
            "errors": errors,
            "strategies": strategies,
            "channels": [
                {
                    key: value
                    for key, value in run.items()
                    if key != "items"
                }
                for run in sorted(channel_runs, key=lambda item: item.get("channel", ""))
            ],
            "channel_summary": channel_summary,
            "pipeline_nodes": self._build_pipeline_nodes(
                understanding,
                channel_runs,
                collected,
                deduped,
                fused,
                items,
            ),
            "sources": sources,
        }
        return RetrievalEngineResult(items=items, context=context, sources=sources, trace=trace)

    async def _run_channel(self, channel: str, awaitable: Any) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            items = await awaitable
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "channel": channel,
                "status": "success",
                "item_count": len(items),
                "latency_ms": latency_ms,
                "items": items,
                "top_sources": [
                    {
                        "source": item.source,
                        "query": item.query,
                        "rank": item.rank,
                        "score": item.score,
                    }
                    for item in items[:5]
                ],
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warning("检索通道失败: channel={}, error={}", channel, exc)
            return {
                "channel": channel,
                "status": "error",
                "item_count": 0,
                "latency_ms": latency_ms,
                "items": [],
                "error": str(exc),
                "top_sources": [],
            }

    async def _merge_channel_tasks(self, tasks: list[asyncio.Task[list[RetrievalItem]]]) -> list[RetrievalItem]:
        merged: list[RetrievalItem] = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                logger.warning("Global vector sub-query failed: {}", result)
                continue
            merged.extend(result)
        return merged

    async def _vector_search(
        self,
        query: str,
        kind: str,
        rank_offset: int = 0,
        source_patterns: list[str] | None = None,
        exclude_source_patterns: list[str] | None = None,
    ) -> list[RetrievalItem]:
        fetch_k = config.rag_prefetch_top_k_per_query
        if source_patterns or exclude_source_patterns:
            fetch_k = max(fetch_k * 4, config.rerank_top_n * 2)
        docs = await asyncio.to_thread(
            vector_store_manager.similarity_search,
            query,
            fetch_k,
        )
        if source_patterns:
            docs = [doc for doc in docs if self._doc_matches_source_patterns(doc, source_patterns)]
        if exclude_source_patterns:
            docs = [doc for doc in docs if not self._doc_matches_source_patterns(doc, exclude_source_patterns)]
        docs = docs[: config.rag_prefetch_top_k_per_query]
        items: list[RetrievalItem] = []
        for idx, doc in enumerate(docs):
            items.append(self._doc_to_item(doc, query=query, kind=kind, rank=rank_offset + idx))
        return items

    async def _keyword_search(self, search_queries: list[str]) -> list[RetrievalItem]:
        hits: list[KeywordSearchHit] = []
        for query in search_queries:
            hits.extend(
                await asyncio.to_thread(
                    keyword_search_service.search,
                    query,
                    config.keyword_search_top_k,
                )
            )

        items: list[RetrievalItem] = []
        for idx, hit in enumerate(hits):
            items.append(
                RetrievalItem(
                    kind="keyword",
                    query=hit.query,
                    content=hit.content,
                    source=hit.source,
                    metadata=hit.metadata,
                    rank=idx,
                    score=hit.score,
                )
            )
        return items

    async def _tool_context(self, understanding: QueryUnderstandingResult) -> list[RetrievalItem]:
        try:
            payload = await asyncio.to_thread(query_prometheus_alerts.invoke, {})
        except Exception as e:
            payload = json.dumps(
                {"success": False, "error": str(e), "message": "Prometheus 告警预取失败"},
                ensure_ascii=False,
            )

        return [
            RetrievalItem(
                kind="tool_context",
                query=understanding.rewritten_question,
                content=str(payload),
                source="query_prometheus_alerts",
                rank=10_000,
            )
        ]

    def _doc_to_item(self, doc: Document, query: str, kind: str, rank: int) -> RetrievalItem:
        metadata = dict(doc.metadata or {})
        source = metadata.get("_file_name") or metadata.get("source") or metadata.get("_source") or "unknown"
        headers = [str(metadata.get(key, "")) for key in ("h1", "h2", "h3") if metadata.get(key)]
        if headers:
            source = f"{source} > {' > '.join(headers)}"

        return RetrievalItem(
            kind=kind,
            query=query,
            content=doc.page_content,
            source=str(source),
            metadata=metadata,
            rank=rank,
        )

    def _doc_matches_source_patterns(self, doc: Document, patterns: list[str]) -> bool:
        metadata = dict(doc.metadata or {})
        fields = [
            metadata.get("_file_name", ""),
            metadata.get("source", ""),
            metadata.get("_source", ""),
            metadata.get("h1", ""),
            metadata.get("h2", ""),
            metadata.get("h3", ""),
            doc.page_content[:500],
        ]
        haystack = " ".join(str(field).lower() for field in fields if field)
        return any(str(pattern).lower() in haystack for pattern in patterns if str(pattern).strip())

    def _dedupe_items(self, items: list[RetrievalItem]) -> list[RetrievalItem]:
        priority = {
            "intent_vector": 0,
            "keyword": 1,
            "supplement_vector": 2,
            "global_vector": 3,
            "tool_context": 4,
        }
        seen: dict[str, RetrievalItem] = {}
        deduped: list[RetrievalItem] = []

        for item in sorted(items, key=lambda it: (priority.get(it.kind, 99), it.rank)):
            key = self._item_key(item)
            item.metadata.setdefault("retrieval_channels", [item.kind])
            item.metadata.setdefault("channel_ranks", {item.kind: item.rank})
            if key in seen:
                existing = seen[key]
                channels = existing.metadata.setdefault("retrieval_channels", [])
                if item.kind not in channels:
                    channels.append(item.kind)
                ranks = existing.metadata.setdefault("channel_ranks", {})
                ranks[item.kind] = min(int(ranks.get(item.kind, item.rank)), int(item.rank))
                continue
            seen[key] = item
            deduped.append(item)

        return deduped[: config.rerank_candidate_limit]

    def _fuse_by_rrf(
        self,
        items: list[RetrievalItem],
        channel_runs: list[dict[str, Any]],
    ) -> list[RetrievalItem]:
        if not config.enable_rrf_fusion or not items:
            return items[: config.rerank_candidate_limit]

        weights = {
            "intent_vector": 1.25,
            "keyword": 1.15,
            "supplement_vector": 0.9,
            "global_vector": 1.0,
            "tool_context": 0.75,
        }
        rrf_scores: dict[str, float] = {}
        channels_by_key: dict[str, set[str]] = {}
        ranks_by_key: dict[str, dict[str, int]] = {}

        for run in channel_runs:
            channel = str(run.get("channel") or "unknown")
            weight = weights.get(channel, 1.0)
            for rank, item in enumerate(run.get("items", []), start=1):
                key = self._item_key(item)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (config.rrf_k + rank)
                channels_by_key.setdefault(key, set()).add(channel)
                ranks_by_key.setdefault(key, {})[channel] = rank

        fused = list(items)
        for item in fused:
            key = self._item_key(item)
            channels = sorted(channels_by_key.get(key) or set(item.metadata.get("retrieval_channels", [item.kind])))
            item.metadata["rrf_score"] = round(rrf_scores.get(key, 0.0), 6)
            item.metadata["rrf_channels"] = channels
            item.metadata["rrf_channel_count"] = len(channels)
            item.metadata["channel_ranks"] = ranks_by_key.get(key, item.metadata.get("channel_ranks", {}))

        fused.sort(
            key=lambda item: (
                float(item.metadata.get("rrf_score", 0.0)),
                int(item.metadata.get("rrf_channel_count", 1)),
                -int(item.rank),
            ),
            reverse=True,
        )
        return fused[: config.rerank_candidate_limit]

    def _build_channel_summary(
        self,
        channel_runs: list[dict[str, Any]],
        final_items: list[RetrievalItem],
    ) -> dict[str, Any]:
        final_counts: dict[str, int] = {}
        for item in final_items:
            channels = item.metadata.get("rrf_channels") or item.metadata.get("retrieval_channels") or [item.kind]
            for channel in channels:
                final_counts[channel] = final_counts.get(channel, 0) + 1

        return {
            "raw_by_channel": {
                str(run.get("channel")): int(run.get("item_count", 0))
                for run in channel_runs
            },
            "final_by_channel": final_counts,
            "latency_by_channel_ms": {
                str(run.get("channel")): float(run.get("latency_ms", 0))
                for run in channel_runs
            },
        }

    def _build_pipeline_nodes(
        self,
        understanding: QueryUnderstandingResult,
        channel_runs: list[dict[str, Any]],
        collected: list[RetrievalItem],
        deduped: list[RetrievalItem],
        fused: list[RetrievalItem],
        final_items: list[RetrievalItem],
    ) -> list[dict[str, Any]]:
        channel_latency = sum(float(run.get("latency_ms", 0)) for run in channel_runs)
        return [
            {
                "name": "query_understanding",
                "type": "UNDERSTAND",
                "status": "success",
                "summary": f"{understanding.intent} / {understanding.confidence:.2f}",
                "input": understanding.original_question,
                "output": understanding.rewritten_question,
                "details": {
                    "normalized_question": understanding.normalized_question,
                    "term_mappings": understanding.applied_mappings,
                    "search_queries": understanding.search_queries,
                    "sub_questions": understanding.sub_questions,
                    "intent_nodes": understanding.intent_nodes,
                    "guidance_prompt": understanding.guidance_prompt,
                    "retrieval_scope": understanding.retrieval_scope,
                    "rewrite_strategy": understanding.rewrite_strategy,
                    "confidence_sources": understanding.confidence_sources,
                    "missing_slots": understanding.missing_slots,
                },
            },
            {
                "name": "parallel_retrieval",
                "type": "RETRIEVE_CHANNEL",
                "status": "success" if collected else "empty",
                "summary": f"{len(channel_runs)} channels / {len(collected)} raw chunks",
                "duration_ms": round(channel_latency, 2),
                "details": [
                    {
                        key: value
                        for key, value in run.items()
                        if key != "items"
                    }
                    for run in channel_runs
                ],
            },
            {
                "name": "deduplication",
                "type": "POST_PROCESS",
                "status": "success",
                "summary": f"{len(collected)} -> {len(deduped)}",
            },
            {
                "name": "rrf_fusion",
                "type": "POST_PROCESS",
                "status": "success" if config.enable_rrf_fusion else "disabled",
                "summary": f"{len(deduped)} -> {len(fused)} candidates",
                "details": {
                    "rrf_k": config.rrf_k,
                    "top_candidates": [
                        {
                            "source": item.source,
                            "kind": item.kind,
                            "rrf_score": item.metadata.get("rrf_score"),
                            "channels": item.metadata.get("rrf_channels"),
                        }
                        for item in fused[:5]
                    ],
                },
            },
            {
                "name": "rerank",
                "type": "RERANK",
                "status": "success" if config.enable_rerank else "disabled",
                "summary": f"{len(fused)} -> {len(final_items)} final contexts",
                "details": {
                    "semantic": config.enable_semantic_rerank,
                    "strategy": rerank_service.strategy_name(),
                    "top_contexts": [
                        {
                            "source": item.source,
                            "kind": item.kind,
                            "score": item.metadata.get("rerank_score", item.score),
                            "final_rank": item.metadata.get("final_rank"),
                        }
                        for item in final_items[:5]
                    ],
                },
            },
        ]

    def _annotate_final_ranks(self, items: list[RetrievalItem]) -> None:
        for idx, item in enumerate(items, start=1):
            item.metadata["final_rank"] = idx

    def _item_key(self, item: RetrievalItem) -> str:
        normalized_content = " ".join(str(item.content or "").split())[:240]
        return f"{item.source}:{normalized_content}"

    def _build_sources(self, items: list[RetrievalItem]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            metadata = item.metadata or {}
            sources.append(
                {
                    "id": f"S{idx}",
                    "kind": item.kind,
                    "source": item.source or "unknown",
                    "query": item.query,
                    "score": item.score or metadata.get("rerank_score") or metadata.get("retrieval_score"),
                    "rerank_score": metadata.get("rerank_score"),
                    "rrf_score": metadata.get("rrf_score"),
                    "channels": metadata.get("rrf_channels") or metadata.get("retrieval_channels") or [item.kind],
                    "final_rank": metadata.get("final_rank", idx),
                    "file_name": metadata.get("_file_name", ""),
                    "excerpt": " ".join(str(item.content or "").split())[:140],
                }
            )
        return sources

    def _format_context(self, items: list[RetrievalItem]) -> str:
        if not items:
            return ""

        parts: list[str] = []
        for idx, item in enumerate(items, start=1):
            parts.append(
                "\n".join(
                    [
                        f"【检索结果 S{idx}】",
                        f"类型: {item.kind}",
                        f"来源: {item.source or 'unknown'}",
                        f"查询: {item.query}",
                        f"重排分数: {item.metadata.get('rerank_score', item.score)}",
                        "内容:",
                        item.content,
                    ]
                )
            )
        return "\n\n".join(parts)


retrieval_engine_service = RetrievalEngineService()
