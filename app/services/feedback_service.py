"""Feedback and lightweight RAG evaluation service."""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from loguru import logger

from app.config import config
from app.services.query_understanding_service import query_understanding_service
from app.services.retrieval_engine_service import RetrievalItem, retrieval_engine_service
from app.services.vector_store_manager import vector_store_manager


@dataclass
class FeedbackRecord:
    session_id: str
    message_id: str = ""
    vote: int = 0
    comment: str = ""
    question: str = ""
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalEvalSample:
    id: str
    category: str
    question: str
    expected_sources: tuple[str, ...]
    expected_intent: str = "aiops_diagnosis"
    setup_turns: tuple[tuple[str, str], ...] = ()
    difficulty: str = "medium"
    tags: tuple[str, ...] = ()


class FeedbackService:
    """Persist user feedback and run seed retrieval evaluations."""

    def __init__(self) -> None:
        self.logs_dir = Path("logs")
        self.feedback_file = self.logs_dir / "rag_feedback.jsonl"
        self.eval_file = self.logs_dir / "rag_eval_runs.jsonl"
        self.eval_dataset_path = Path("data") / "eval" / "rag_eval_cases.json"
        self.eval_samples, self.eval_dataset_version = self._load_eval_samples()

    def record_feedback(self, record: FeedbackRecord) -> dict[str, Any]:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **asdict(record),
        }
        self._append(self.feedback_file, payload)
        return payload

    def get_retrieval_eval_metadata(self) -> dict[str, Any]:
        """Return dataset metadata without running retrieval or embedding calls."""
        categories = Counter(sample.category for sample in self.eval_samples)
        difficulties = Counter(sample.difficulty for sample in self.eval_samples)
        tags = Counter(tag for sample in self.eval_samples for tag in sample.tags)
        return {
            "version": self.eval_dataset_version,
            "dataset_path": str(self.eval_dataset_path),
            "sample_count": len(self.eval_samples),
            "categories": dict(categories.most_common()),
            "difficulties": dict(difficulties.most_common()),
            "top_tags": dict(tags.most_common(12)),
            "preview": [
                {
                    "id": sample.id,
                    "category": sample.category,
                    "difficulty": sample.difficulty,
                    "question": sample.question,
                    "expected_intent": sample.expected_intent,
                    "expected_sources": list(sample.expected_sources),
                }
                for sample in self.eval_samples[:8]
            ],
        }

    async def run_retrieval_eval(self) -> dict[str, Any]:
        """Compare baseline vector retrieval with the enhanced RAG pipeline."""
        results: list[dict[str, Any]] = []

        for idx, sample in enumerate(self.eval_samples, start=1):
            session_id = f"eval-seed-{idx}"
            query_understanding_service.clear_memory(session_id)
            self._prime_memory(session_id, sample)

            if config.enable_llm_retrieval_eval:
                understanding = await query_understanding_service.analyze_async(
                    sample.question,
                    session_id,
                    user_id="eval",
                )
            else:
                understanding = query_understanding_service.analyze(sample.question, session_id, user_id="eval")
            enhanced = await retrieval_engine_service.retrieve(understanding)
            baseline_items = await self._run_baseline_vector_search(sample.question)

            expected = [item.lower() for item in sample.expected_sources]
            enhanced_sources = self._source_records(enhanced.items)
            baseline_sources = self._source_records(baseline_items)
            enhanced_rank = self._first_hit_rank(enhanced_sources, expected)
            baseline_rank = self._first_hit_rank(baseline_sources, expected)

            results.append(
                {
                    "id": sample.id,
                    "category": sample.category,
                    "question": sample.question,
                    "expected_sources": list(sample.expected_sources),
                    "expected_intent": sample.expected_intent,
                    "difficulty": sample.difficulty,
                    "tags": list(sample.tags),
                    "predicted_intent": understanding.intent,
                    "intent_hit": understanding.intent == sample.expected_intent,
                    "confidence": understanding.confidence,
                    "rewritten_question": understanding.rewritten_question,
                    "rewrite_strategy": understanding.rewrite_strategy,
                    "search_queries": understanding.search_queries,
                    "sub_questions": understanding.sub_questions,
                    "intent_nodes": understanding.intent_nodes[:3],
                    "retrieval_scope": understanding.retrieval_scope,
                    "baseline": self._case_metrics(baseline_rank, baseline_sources),
                    "enhanced": self._case_metrics(enhanced_rank, enhanced_sources),
                    "trace": enhanced.trace,
                }
            )

        baseline_metrics = self._aggregate_metrics([item["baseline"] for item in results])
        enhanced_metrics = self._aggregate_metrics([item["enhanced"] for item in results])
        intent_accuracy = mean(1 if item["intent_hit"] else 0 for item in results) if results else 0.0
        hit_count = sum(1 for item in results if item["enhanced"]["hit_at_5"])

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "version": self.eval_dataset_version,
            "dataset_path": str(self.eval_dataset_path),
            "sample_count": len(results),
            "hit_count": hit_count,
            "hit_rate": enhanced_metrics["hit_at_5"],
            "baseline": baseline_metrics,
            "enhanced": enhanced_metrics,
            "improvement": self._diff_metrics(baseline_metrics, enhanced_metrics),
            "intent_accuracy": round(intent_accuracy, 4),
            "metric_notes": {
                "hit_at_1": "正确文档出现在第 1 个结果中的比例。",
                "hit_at_3": "正确文档出现在前 3 个结果中的比例。",
                "hit_at_5": "正确文档出现在前 5 个结果中的比例。",
                "mrr": "Mean Reciprocal Rank，正确文档越靠前分数越高。",
                "ndcg_at_5": "Normalized Discounted Cumulative Gain，衡量前 5 个结果的排序收益。",
                "baseline": "原始问题直接做向量检索。",
                "enhanced": "意图识别、问题重写、Query 扩展、混合检索和重排后的结果。",
            },
            "results": results,
        }
        self._append(self.eval_file, payload)
        return payload

    def _prime_memory(self, session_id: str, sample: RetrievalEvalSample) -> None:
        for question, answer in sample.setup_turns:
            understanding = query_understanding_service.analyze(question, session_id, user_id="eval")
            query_understanding_service.update_memory(
                session_id,
                question,
                answer,
                understanding,
                user_id="eval",
                persist_long_term=False,
            )

    async def _run_baseline_vector_search(self, question: str) -> list[RetrievalItem]:
        docs = await asyncio.to_thread(vector_store_manager.similarity_search, question, 5)
        items: list[RetrievalItem] = []
        for idx, doc in enumerate(docs):
            metadata = dict(doc.metadata or {})
            source = metadata.get("_file_name") or metadata.get("source") or metadata.get("_source") or "unknown"
            items.append(
                RetrievalItem(
                    kind="baseline_vector",
                    query=question,
                    content=doc.page_content,
                    source=str(source),
                    metadata=metadata,
                    rank=idx,
                )
            )
        return items

    def _source_records(self, items: list[RetrievalItem]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(items[:5], start=1):
            metadata = item.metadata or {}
            records.append(
                {
                    "rank": idx,
                    "kind": item.kind,
                    "source": item.source or "unknown",
                    "file_name": metadata.get("_file_name", ""),
                    "score": item.score or metadata.get("rerank_score") or metadata.get("retrieval_score"),
                    "rerank_score": metadata.get("rerank_score"),
                    "rrf_score": metadata.get("rrf_score"),
                    "channels": metadata.get("rrf_channels") or metadata.get("retrieval_channels") or [item.kind],
                }
            )
        return records

    def _first_hit_rank(self, sources: list[dict[str, Any]], expected_sources: list[str]) -> int | None:
        for source in sources:
            haystack = " ".join(
                str(source.get(key, "")).lower()
                for key in ("source", "file_name", "kind")
            )
            if any(expected in haystack for expected in expected_sources):
                return int(source["rank"])
        return None

    def _case_metrics(self, hit_rank: int | None, sources: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "hit_rank": hit_rank,
            "hit_at_1": bool(hit_rank and hit_rank <= 1),
            "hit_at_3": bool(hit_rank and hit_rank <= 3),
            "hit_at_5": bool(hit_rank and hit_rank <= 5),
            "reciprocal_rank": round(1 / hit_rank, 4) if hit_rank else 0.0,
            "ndcg_at_5": round(1 / math.log2(hit_rank + 1), 4) if hit_rank and hit_rank <= 5 else 0.0,
            "sources": sources,
        }

    def _aggregate_metrics(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        total = max(1, len(cases))
        return {
            "hit_at_1": round(sum(1 for case in cases if case["hit_at_1"]) / total, 4),
            "hit_at_3": round(sum(1 for case in cases if case["hit_at_3"]) / total, 4),
            "hit_at_5": round(sum(1 for case in cases if case["hit_at_5"]) / total, 4),
            "mrr": round(sum(float(case["reciprocal_rank"]) for case in cases) / total, 4),
            "ndcg_at_5": round(sum(float(case["ndcg_at_5"]) for case in cases) / total, 4),
        }

    def _diff_metrics(self, baseline: dict[str, Any], enhanced: dict[str, Any]) -> dict[str, Any]:
        return {
            key: round(float(enhanced.get(key, 0.0)) - float(baseline.get(key, 0.0)), 4)
            for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5")
        }

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _load_eval_samples(self) -> tuple[list[RetrievalEvalSample], str]:
        """Load retrieval eval samples from JSON, with built-in samples as fallback."""
        if not self.eval_dataset_path.exists():
            return self._build_seed_eval_samples(), "seed-v1"

        try:
            payload = json.loads(self.eval_dataset_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_cases = payload.get("cases", [])
                version = str(payload.get("version") or "external-v1")
            else:
                raw_cases = payload
                version = "external-v1"

            samples = [self._sample_from_payload(case) for case in raw_cases if isinstance(case, dict)]
            samples = [sample for sample in samples if sample.question and sample.expected_sources]
            if not samples:
                raise ValueError("no valid retrieval eval cases found")

            logger.info(
                "Loaded retrieval eval dataset: path={}, version={}, samples={}",
                self.eval_dataset_path,
                version,
                len(samples),
            )
            return samples, version
        except Exception as exc:
            logger.warning(
                "Failed to load retrieval eval dataset from {}, fallback to seed samples: {}",
                self.eval_dataset_path,
                exc,
            )
            return self._build_seed_eval_samples(), "seed-v1"

    def _sample_from_payload(self, payload: dict[str, Any]) -> RetrievalEvalSample:
        setup_turns = []
        for turn in payload.get("setup_turns", []) or []:
            if isinstance(turn, dict):
                question = str(turn.get("question", "")).strip()
                answer = str(turn.get("answer", "")).strip()
            elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                question = str(turn[0]).strip()
                answer = str(turn[1]).strip()
            else:
                continue
            if question and answer:
                setup_turns.append((question, answer))

        return RetrievalEvalSample(
            id=str(payload.get("id", "")).strip(),
            category=str(payload.get("category", "未分类")).strip() or "未分类",
            question=str(payload.get("question", "")).strip(),
            expected_sources=tuple(str(item).strip() for item in payload.get("expected_sources", []) if str(item).strip()),
            expected_intent=str(payload.get("expected_intent", "aiops_diagnosis")).strip() or "aiops_diagnosis",
            setup_turns=tuple(setup_turns),
            difficulty=str(payload.get("difficulty", "medium")).strip() or "medium",
            tags=tuple(str(item).strip() for item in payload.get("tags", []) if str(item).strip()),
        )

    def _build_seed_eval_samples(self) -> list[RetrievalEvalSample]:
        return [
            RetrievalEvalSample(
                id="cpu-basic",
                category="CPU 故障",
                question="CPU 过高怎么排查？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="cpu-java-thread",
                category="CPU 故障",
                question="Java 服务 CPU 飙高，怎么定位热点线程？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="memory-leak",
                category="内存故障",
                question="内存泄漏应该怎么定位？",
                expected_sources=("memory_high_usage", "memory"),
            ),
            RetrievalEvalSample(
                id="disk-full",
                category="磁盘故障",
                question="磁盘空间快满了怎么处理？",
                expected_sources=("disk_high_usage", "disk"),
            ),
            RetrievalEvalSample(
                id="slow-response",
                category="性能故障",
                question="服务响应慢如何排查？",
                expected_sources=("slow_response", "slow"),
            ),
            RetrievalEvalSample(
                id="service-unavailable",
                category="可用性故障",
                question="服务不可用怎么恢复？",
                expected_sources=("service_unavailable", "service"),
            ),
            RetrievalEvalSample(
                id="redis-interview",
                category="Redis 知识",
                question="Redis 持久化和高可用面试怎么回答？",
                expected_sources=("redis", "Redis_面试模拟"),
                expected_intent="knowledge_query",
            ),
            RetrievalEvalSample(
                id="redis-rdb-aof",
                category="Redis 知识",
                question="RDB 和 AOF 有什么区别？",
                expected_sources=("redis", "Redis_面试模拟"),
                expected_intent="knowledge_query",
            ),
            RetrievalEvalSample(
                id="follow-up-java-thread",
                category="多轮追问",
                question="那如果是 Java 怎么看线程？",
                expected_sources=("cpu_high_usage", "cpu"),
                setup_turns=(
                    (
                        "CPU 过高怎么排查？",
                        "需要先定位高 CPU 进程，再结合线程栈分析热点线程。",
                    ),
                ),
            ),
            RetrievalEvalSample(
                id="cpu-mitigation",
                category="CPU 故障",
                question="CPU 告警触发后临时止血怎么做？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="cpu-process",
                category="CPU 故障",
                question="单个进程 CPU 打满应该先看什么？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="cpu-loop",
                category="CPU 故障",
                question="怎么判断 CPU 高是不是死循环导致的？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="cpu-release",
                category="CPU 故障",
                question="发布后 CPU 突然升高怎么排查？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="memory-high",
                category="内存故障",
                question="内存使用率过高时应该怎么排查？",
                expected_sources=("memory_high_usage", "memory"),
            ),
            RetrievalEvalSample(
                id="memory-oom",
                category="内存故障",
                question="应用出现 OOM 应该保留哪些现场？",
                expected_sources=("memory_high_usage", "memory"),
            ),
            RetrievalEvalSample(
                id="memory-jvm-dump",
                category="内存故障",
                question="Java 服务怀疑内存泄漏，怎么生成 heap dump？",
                expected_sources=("memory_high_usage", "memory"),
            ),
            RetrievalEvalSample(
                id="memory-gc",
                category="内存故障",
                question="频繁 GC 和内存过高有关吗，怎么查？",
                expected_sources=("memory_high_usage", "memory"),
            ),
            RetrievalEvalSample(
                id="disk-large-files",
                category="磁盘故障",
                question="磁盘满了怎么快速找大文件？",
                expected_sources=("disk_high_usage", "disk"),
            ),
            RetrievalEvalSample(
                id="disk-inode",
                category="磁盘故障",
                question="磁盘空间没满但文件创建失败，inode 怎么检查？",
                expected_sources=("disk_high_usage", "disk"),
            ),
            RetrievalEvalSample(
                id="disk-log-clean",
                category="磁盘故障",
                question="日志文件太大导致磁盘告警怎么处理？",
                expected_sources=("disk_high_usage", "disk"),
            ),
            RetrievalEvalSample(
                id="disk-readonly",
                category="磁盘故障",
                question="文件系统变成只读和磁盘故障怎么排查？",
                expected_sources=("disk_high_usage", "disk"),
            ),
            RetrievalEvalSample(
                id="slow-timeout",
                category="性能故障",
                question="接口请求超时应该从哪些方面排查？",
                expected_sources=("slow_response", "slow"),
            ),
            RetrievalEvalSample(
                id="slow-db",
                category="性能故障",
                question="服务响应慢可能是数据库慢查询导致的吗？",
                expected_sources=("slow_response", "slow"),
            ),
            RetrievalEvalSample(
                id="slow-thread-pool",
                category="性能故障",
                question="线程池耗尽会导致响应慢吗？",
                expected_sources=("slow_response", "slow"),
            ),
            RetrievalEvalSample(
                id="slow-traffic",
                category="性能故障",
                question="流量突增导致响应变慢怎么定位？",
                expected_sources=("slow_response", "slow"),
            ),
            RetrievalEvalSample(
                id="service-health",
                category="可用性故障",
                question="服务不可用时先检查健康检查还是日志？",
                expected_sources=("service_unavailable", "service"),
            ),
            RetrievalEvalSample(
                id="service-dependency",
                category="可用性故障",
                question="下游依赖异常导致服务不可用怎么排查？",
                expected_sources=("service_unavailable", "service"),
            ),
            RetrievalEvalSample(
                id="service-rollback",
                category="可用性故障",
                question="服务发布后不可用，什么时候应该回滚？",
                expected_sources=("service_unavailable", "service"),
            ),
            RetrievalEvalSample(
                id="service-restart",
                category="可用性故障",
                question="服务挂了以后重启前要确认哪些信息？",
                expected_sources=("service_unavailable", "service"),
            ),
            RetrievalEvalSample(
                id="redis-cache-avalanche",
                category="Redis 知识",
                question="Redis 缓存雪崩、击穿、穿透有什么区别？",
                expected_sources=("redis", "Redis_面试模拟"),
                expected_intent="knowledge_query",
            ),
            RetrievalEvalSample(
                id="redis-lock",
                category="Redis 知识",
                question="Redis 分布式锁面试怎么讲？",
                expected_sources=("redis", "Redis_面试模拟"),
                expected_intent="knowledge_query",
            ),
            RetrievalEvalSample(
                id="redis-memory",
                category="Redis 知识",
                question="Redis 过期策略和内存淘汰怎么回答？",
                expected_sources=("redis", "Redis_面试模拟"),
                expected_intent="knowledge_query",
            ),
            RetrievalEvalSample(
                id="follow-up-memory-dump",
                category="多轮追问",
                question="那 dump 文件怎么拿？",
                expected_sources=("memory_high_usage", "memory"),
                setup_turns=(
                    (
                        "内存泄漏应该怎么定位？",
                        "需要观察内存趋势，保留现场，并通过 heap dump 或内存分析工具定位泄漏对象。",
                    ),
                ),
            ),
            RetrievalEvalSample(
                id="follow-up-disk-clean",
                category="多轮追问",
                question="那临时怎么清理？",
                expected_sources=("disk_high_usage", "disk"),
                setup_turns=(
                    (
                        "磁盘空间快满了怎么处理？",
                        "需要先找大文件和日志增长点，再做安全清理或扩容。",
                    ),
                ),
            ),
            RetrievalEvalSample(
                id="vague-cpu",
                category="模糊问法",
                question="机器负载突然很高怎么办？",
                expected_sources=("cpu_high_usage", "cpu"),
            ),
            RetrievalEvalSample(
                id="vague-service",
                category="模糊问法",
                question="用户说系统打不开了，我该怎么查？",
                expected_sources=("service_unavailable", "service"),
            ),
        ]


feedback_service = FeedbackService()
