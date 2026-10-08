"""查询理解服务：意图识别、问题改写和增强会话记忆。

这一层位于 API/Agent 之间，作用是把用户的原始输入变成更适合 Agent
理解和检索的上下文，同时维护一个轻量的会话级摘要记忆。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from loguru import logger

from app.config import config
from app.services.intent_tree_service import intent_tree_service
from app.services.llm_intent_classifier_service import llm_intent_classifier_service
from app.services.long_term_memory_service import long_term_memory_service
from app.services.model_router_service import model_router_service
from app.services.query_rewrite_service import query_rewrite_service
from app.services.term_mapping_service import term_mapping_service


IntentType = Literal[
    "aiops_diagnosis",
    "knowledge_query",
    "metrics_alerts",
    "time_query",
    "general_chat",
    "unknown",
]


@dataclass
class QueryUnderstandingResult:
    """单次用户输入的理解结果。"""

    original_question: str
    intent: IntentType
    confidence: float
    rewritten_question: str
    search_queries: list[str]
    sub_questions: list[str] = field(default_factory=list)
    normalized_question: str = ""
    applied_mappings: list[dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""
    route_hint: str = ""
    reasons: list[str] = field(default_factory=list)
    confidence_sources: dict[str, Any] = field(default_factory=dict)
    intent_nodes: list[dict[str, Any]] = field(default_factory=list)
    guidance_prompt: str = ""
    retrieval_scope: dict[str, Any] = field(default_factory=dict)
    rewrite_strategy: str = "rule_rewrite"
    missing_slots: list[str] = field(default_factory=list)
    related_memory: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典，便于日志/API 调试。"""
        return asdict(self)


@dataclass
class SessionMemory:
    """增强会话记忆。

    MemorySaver 已经保存完整消息历史；这里额外维护“摘要 + 稳定事实 +
    最近轮次”，用于在长对话里给 Agent 一个更紧凑的上下文。
    """

    summary: str = ""
    last_topic: str = ""
    stable_facts: list[str] = field(default_factory=list)
    topic_stack: list[str] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, list[str]] = field(default_factory=dict)
    focus_questions: list[str] = field(default_factory=list)
    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryUnderstandingService:
    """轻量查询理解服务。

    当前实现优先使用确定性规则，避免每次请求额外调用一次 LLM。
    后续如果希望更智能，可以在本服务中替换为 LLM structured output，
    但对外接口不需要变化。
    """

    FOLLOW_UP_MARKERS = (
        "这个",
        "这个项目",
        "上面",
        "刚才",
        "前面",
        "继续",
        "详细讲",
        "详细说",
        "展开讲",
        "它",
        "这块",
        "这一块",
        "那这个",
        "那",
        "后面",
        "以后",
        "接下来",
        "下一步",
        "为什么",
        "怎么做",
    )

    TIME_KEYWORDS = ("几点", "日期", "今天", "明天", "昨天", "星期", "时间")
    AIOPS_KEYWORDS = (
        "诊断",
        "故障",
        "排查",
        "告警",
        "宕机",
        "服务不可用",
        "响应慢",
        "cpu过高",
        "cpu 高",
        "负载",
        "load",
        "内存过高",
        "内存泄漏",
        "oom",
        "gc",
        "磁盘满",
        "超时",
        "变慢",
        "流量突增",
        "打不开",
        "挂了",
        "回滚",
        "下游",
        "a i ops",
        "aiops",
        "运维",
    )
    METRICS_KEYWORDS = ("prometheus", "firing", "pending", "监控指标", "告警列表", "当前告警")
    KNOWLEDGE_KEYWORDS = (
        "什么是",
        "解释",
        "讲解",
        "原理",
        "实现",
        "代码",
        "面试",
        "rag",
        "mcp",
        "milvus",
        "collection",
        "embedding",
        "langgraph",
        "langchain",
        "redis",
        "rdb",
        "aof",
        "缓存",
        "分布式锁",
        "工具",
    )

    def __init__(self) -> None:
        self._memories: dict[str, SessionMemory] = {}

    async def analyze_async(
        self,
        question: str,
        session_id: str,
        user_id: str | None = None,
    ) -> QueryUnderstandingResult:
        """LLM-assisted query understanding with deterministic fallback."""
        result = self.analyze(question, session_id, user_id=user_id)

        if config.enable_llm_query_understanding:
            if not config.dashscope_api_key:
                result.reasons.append("llm_query_rewrite_skipped: empty_dashscope_api_key")
            else:
                try:
                    payload = await asyncio.wait_for(
                        query_rewrite_service.rewrite(question, result.to_dict()),
                        timeout=max(1.0, config.llm_query_understanding_timeout_seconds),
                    )
                    result = self._merge_llm_understanding(result, payload)
                except Exception as exc:
                    logger.warning("LLM 查询理解失败，回退到规则结果: {}", exc)
                    result.reasons.append(f"llm_query_understanding_fallback: {self._compact_llm_error(exc)}")

        if config.enable_llm_intent_tree_scoring:
            result = await self._apply_llm_intent_tree(result, session_id)

        return result

    def analyze(
        self,
        question: str,
        session_id: str,
        user_id: str | None = None,
    ) -> QueryUnderstandingResult:
        """识别意图、改写问题并注入会话记忆。"""
        cleaned_question = self._normalize_question(question)
        normalized_question, applied_mappings = term_mapping_service.normalize(cleaned_question)
        normalized_question = self._normalize_question(normalized_question)
        memory = self._memories.get(session_id, SessionMemory())
        related_turns = self._retrieve_relevant_turns(memory, normalized_question)
        long_term_memory = long_term_memory_service.get_user_memory(user_id)
        long_term_topic = self._latest_long_term_topic(long_term_memory)

        if not config.enable_query_understanding:
            long_term_context = long_term_memory_service.format_context(
                user_id,
                intent="unknown",
                allow_topic_recall=False,
            )
            return QueryUnderstandingResult(
                original_question=question,
                intent="unknown",
                confidence=0.0,
                rewritten_question=normalized_question,
                search_queries=[normalized_question],
                sub_questions=[normalized_question],
                normalized_question=normalized_question,
                applied_mappings=applied_mappings,
                memory_context=self._format_memory_context(
                    memory,
                    normalized_question,
                    related_turns,
                    long_term_context,
                    intent="unknown",
                    include_topic_context=False,
                ),
                route_hint="查询理解功能已关闭，直接使用原始问题。",
                reasons=["enable_query_understanding=false"],
                retrieval_scope={"mode": "global", "reason": "query_understanding_disabled"},
                rewrite_strategy="disabled",
                related_memory=related_turns,
            )

        intent, confidence, reasons = self._classify_intent(normalized_question)
        rule_intent = intent
        rule_confidence = confidence
        memory_confidence_floor = 0.0
        if applied_mappings:
            reasons.append(
                "命中术语映射: "
                + "；".join(
                    f"{item.get('source_term')} -> {item.get('target_term')}"
                    for item in applied_mappings[:4]
                )
            )
        if memory.last_topic and self._looks_like_follow_up(normalized_question):
            topic_lower = memory.last_topic.lower()
            if any(marker in topic_lower for marker in ("aiops", "运维", "故障", "诊断", "cpu", "负载", "内存", "磁盘", "服务")):
                intent = "aiops_diagnosis"
                confidence = max(confidence, 0.78)
                memory_confidence_floor = max(memory_confidence_floor, 0.78)
                reasons.append("follow-up inherits previous AIOps diagnosis topic")
        allow_long_term_topic_recall = self._should_use_long_term_topic(
            memory,
            normalized_question,
            intent,
            long_term_topic,
        )
        if allow_long_term_topic_recall:
            topic_lower = long_term_topic.lower()
            if any(marker in topic_lower for marker in ("aiops", "运维", "故障", "诊断", "cpu", "内存", "磁盘", "服务")):
                intent = "aiops_diagnosis"
                confidence = max(confidence, 0.72)
                memory_confidence_floor = max(memory_confidence_floor, 0.72)
            reasons.append("follow-up can use relevant long-term user memory")
        long_term_context = long_term_memory_service.format_context(
            user_id,
            intent=intent,
            allow_topic_recall=allow_long_term_topic_recall,
        )
        rewrite_long_term_topic = long_term_topic if allow_long_term_topic_recall else ""
        rewritten_question = self._rewrite_question(
            normalized_question,
            memory,
            intent,
            rewrite_long_term_topic,
        )
        sub_questions = self._split_sub_questions(normalized_question, rewritten_question)
        intent_matches = intent_tree_service.resolve_many(
            [normalized_question, rewritten_question, *sub_questions],
            limit=config.intent_tree_max_candidates,
        )
        intent_nodes = [node.to_dict() for node in intent_matches]
        retrieval_scope = intent_tree_service.build_retrieval_scope(intent_matches)
        guidance_prompt = intent_tree_service.maybe_ambiguity_prompt(normalized_question, intent_matches)
        top_score = 0.0
        tree_route = ""
        tree_path = ""
        if intent_nodes:
            top_node = intent_nodes[0]
            reasons.append(
                f"命中意图树节点: {top_node.get('path') or (str(top_node.get('parent')) + ' > ' + str(top_node.get('name')))}"
            )
            top_score = float(top_node.get("score", 0.0))
            tree_route = str(top_node.get("route") or "")
            tree_path = str(top_node.get("path") or top_node.get("name") or "")
            if confidence < top_score:
                intent = top_node.get("route", intent)  # type: ignore[assignment]
                confidence = top_score
        confidence, confidence_sources = self._fuse_confidence(
            rule_intent=rule_intent,
            rule_confidence=rule_confidence,
            memory_confidence_floor=memory_confidence_floor,
            tree_route=tree_route,
            tree_score=top_score,
            tree_path=tree_path,
            llm_intent=None,
            llm_confidence=None,
            current_intent=intent,
            source="rule_tree",
        )
        if guidance_prompt:
            reasons.append("意图树候选分数接近，触发歧义引导")
        search_queries = self._build_search_queries(
            normalized_question,
            rewritten_question,
            intent,
            memory,
            related_turns,
            sub_questions=sub_questions,
            intent_nodes=intent_nodes,
        )
        long_term_context = long_term_memory_service.format_context(
            user_id,
            intent=intent,
            allow_topic_recall=allow_long_term_topic_recall,
        )
        memory_context = self._format_memory_context(
            memory,
            normalized_question,
            related_turns,
            long_term_context,
            intent=intent,
            include_topic_context=self._looks_like_follow_up(normalized_question) or bool(related_turns),
        )
        route_hint = self._build_route_hint(intent)

        result = QueryUnderstandingResult(
            original_question=question,
            intent=intent,
            confidence=confidence,
            rewritten_question=rewritten_question,
            search_queries=search_queries,
            sub_questions=sub_questions,
            normalized_question=normalized_question,
            applied_mappings=applied_mappings,
            memory_context=memory_context,
            route_hint=route_hint,
            reasons=reasons,
            confidence_sources=confidence_sources,
            intent_nodes=intent_nodes,
            guidance_prompt=guidance_prompt,
            retrieval_scope=retrieval_scope,
            rewrite_strategy="rule_rewrite_and_split",
            related_memory=related_turns,
        )
        logger.info(
            "查询理解完成: session={}, intent={}, confidence={:.2f}, rewritten={}",
            session_id,
            intent,
            confidence,
            rewritten_question,
        )
        return result

    async def _llm_understand(self, question: str, base_result: QueryUnderstandingResult) -> dict[str, Any]:
        llm = model_router_service.create_chat_model(
            config.llm_query_understanding_model,
            streaming=False,
        )
        prompt = f"""
你是一个智能运维 RAG Agent 的查询理解器。请基于用户原始问题、规则基线结果和记忆上下文，输出严格 JSON。

可选 intent:
- aiops_diagnosis: CPU、内存、磁盘、响应慢、服务不可用等排障问题
- knowledge_query: 项目知识、Redis、RAG、MCP、代码实现、面试讲解
- metrics_alerts: Prometheus/告警/监控指标查询
- time_query: 时间日期类问题
- general_chat: 普通聊天
- unknown: 无法判断

要求:
1. 不要输出 Markdown，只输出 JSON。
2. rewritten_question 要补全省略指代，但不能改变用户原意。
3. 如果用户问题包含多个独立问题，sub_questions 拆成 2 到 4 条；否则只包含 rewritten_question。
4. search_queries 给 2 到 4 条，适合知识库/监控检索。
5. confidence 范围 0 到 1。

用户原始问题:
{question}

规则基线:
{json.dumps(base_result.to_dict(), ensure_ascii=False)}

JSON schema:
{{
  "intent": "aiops_diagnosis",
  "confidence": 0.8,
  "rewritten_question": "...",
  "sub_questions": ["..."],
  "search_queries": ["...", "..."],
  "reasons": ["..."]
}}
""".strip()
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", response)
        return self._parse_json_object(str(content))

    def _merge_llm_understanding(
        self,
        base_result: QueryUnderstandingResult,
        payload: dict[str, Any],
    ) -> QueryUnderstandingResult:
        valid_intents = {
            "aiops_diagnosis",
            "knowledge_query",
            "metrics_alerts",
            "time_query",
            "general_chat",
            "unknown",
        }
        intent = str(payload.get("intent") or base_result.intent)
        if intent not in valid_intents:
            intent = base_result.intent

        llm_confidence = payload.get("confidence", None)
        try:
            llm_confidence = max(0.0, min(1.0, float(llm_confidence)))
        except (TypeError, ValueError):
            llm_confidence = None

        rewritten_question = str(payload.get("rewritten_question") or base_result.rewritten_question).strip()
        if not rewritten_question:
            rewritten_question = base_result.rewritten_question

        raw_sub_questions = payload.get("sub_questions") or []
        sub_questions = [
            str(item).strip()
            for item in raw_sub_questions
            if str(item).strip()
        ] or base_result.sub_questions or [rewritten_question]

        raw_queries = payload.get("search_queries") or []
        llm_queries = [
            str(item).strip()
            for item in raw_queries
            if str(item).strip()
        ]
        raw_missing_slots = payload.get("missing_slots") or []
        missing_slots = [
            str(item).strip()
            for item in raw_missing_slots
            if str(item).strip()
        ][:8]
        search_queries = self._dedupe([rewritten_question, *sub_questions, *llm_queries, *base_result.search_queries])[
            : max(1, config.query_rewrite_max_queries)
        ]

        intent_matches = intent_tree_service.resolve_many(
            [base_result.normalized_question, rewritten_question, *sub_questions],
            limit=config.intent_tree_max_candidates,
        )
        intent_nodes = [node.to_dict() for node in intent_matches] or base_result.intent_nodes
        retrieval_scope = intent_tree_service.build_retrieval_scope(intent_matches) if intent_matches else base_result.retrieval_scope
        guidance_prompt = intent_tree_service.maybe_ambiguity_prompt(
            base_result.normalized_question,
            intent_matches,
        ) if intent_matches else base_result.guidance_prompt

        top_node = intent_nodes[0] if intent_nodes else {}
        tree_score = float(top_node.get("score") or 0.0) if top_node else 0.0
        tree_route = str(top_node.get("route") or "") if top_node else ""
        tree_path = str(top_node.get("path") or top_node.get("name") or "") if top_node else ""
        rule_source = base_result.confidence_sources.get("rule", {}) if base_result.confidence_sources else {}
        memory_source = base_result.confidence_sources.get("memory", {}) if base_result.confidence_sources else {}
        confidence, confidence_sources = self._fuse_confidence(
            rule_intent=str(rule_source.get("intent") or base_result.intent),
            rule_confidence=float(rule_source.get("confidence") or base_result.confidence),
            memory_confidence_floor=float(memory_source.get("confidence_floor") or 0.0),
            tree_route=tree_route,
            tree_score=tree_score,
            tree_path=tree_path,
            llm_intent=intent,
            llm_confidence=llm_confidence,
            current_intent=intent,
            source="rule_tree_llm_query",
        )
        llm_reasons = [
            str(item)
            for item in payload.get("reasons", [])
            if str(item).strip()
        ]
        return QueryUnderstandingResult(
            original_question=base_result.original_question,
            intent=intent,  # type: ignore[arg-type]
            confidence=confidence,
            rewritten_question=rewritten_question,
            search_queries=search_queries,
            sub_questions=sub_questions,
            normalized_question=base_result.normalized_question,
            applied_mappings=base_result.applied_mappings,
            memory_context=base_result.memory_context,
            route_hint=self._build_route_hint(intent),  # type: ignore[arg-type]
            reasons=[*base_result.reasons, "llm_query_understanding_applied", *llm_reasons],
            confidence_sources=confidence_sources,
            intent_nodes=intent_nodes,
            guidance_prompt=guidance_prompt,
            retrieval_scope=retrieval_scope,
            rewrite_strategy="llm_rewrite_and_split",
            missing_slots=missing_slots,
            related_memory=base_result.related_memory,
        )

    async def _apply_llm_intent_tree(
        self,
        result: QueryUnderstandingResult,
        session_id: str,
    ) -> QueryUnderstandingResult:
        """Use an LLM to rescore configured intent-tree nodes, with rule fallback."""
        if not config.dashscope_api_key:
            result.reasons.append("llm_intent_tree_skipped: empty_dashscope_api_key")
            return result

        try:
            payload = await asyncio.wait_for(
                llm_intent_classifier_service.classify(
                    question=result.normalized_question or result.original_question,
                    rewritten_question=result.rewritten_question,
                    sub_questions=result.sub_questions,
                    baseline_matches=result.intent_nodes,
                ),
                timeout=max(1.0, config.llm_intent_tree_timeout_seconds),
            )
        except Exception as exc:
            logger.warning("LLM 意图树打分失败，继续使用规则意图树: {}", exc)
            result.reasons.append(f"llm_intent_tree_fallback: {self._compact_llm_error(exc)}")
            return result

        llm_nodes = [
            node
            for node in payload.get("matches", [])
            if isinstance(node, dict)
        ]
        if not llm_nodes:
            result.reasons.append("llm_intent_tree_no_match")
            return result

        result.intent_nodes = llm_nodes
        result.retrieval_scope = intent_tree_service.build_retrieval_scope(
            [
                match
                for node in llm_nodes
                if (
                    match := intent_tree_service.match_from_score(
                        str(node.get("node_id") or node.get("id") or ""),
                        float(node.get("score") or 0.0),
                        reason="llm_intent_tree_scoring",
                        matched_terms=[
                            str(term)
                            for term in node.get("matched_terms", [])
                            if str(term).strip()
                        ],
                    )
                )
            ]
        )
        top_node = llm_nodes[0]
        top_score = float(top_node.get("score") or 0.0)
        if top_score >= config.intent_tree_min_score:
            result.intent = top_node.get("route", result.intent)  # type: ignore[assignment]
            rule_source = result.confidence_sources.get("rule", {}) if result.confidence_sources else {}
            memory_source = result.confidence_sources.get("memory", {}) if result.confidence_sources else {}
            llm_source = result.confidence_sources.get("llm", {}) if result.confidence_sources else {}
            result.confidence, result.confidence_sources = self._fuse_confidence(
                rule_intent=str(rule_source.get("intent") or result.intent),
                rule_confidence=float(rule_source.get("confidence") or result.confidence),
                memory_confidence_floor=float(memory_source.get("confidence_floor") or 0.0),
                tree_route=str(top_node.get("route") or result.intent),
                tree_score=top_score,
                tree_path=str(top_node.get("path") or top_node.get("name") or ""),
                llm_intent=str(llm_source.get("intent") or result.intent) if llm_source else None,
                llm_confidence=llm_source.get("confidence") if llm_source else None,
                current_intent=result.intent,
                source="rule_llm_intent_tree",
                tree_score_source="llm_intent_tree",
            )

        guidance_prompt = str(payload.get("guidance_prompt") or "").strip()
        if payload.get("need_guidance") and guidance_prompt:
            result.guidance_prompt = guidance_prompt
        else:
            llm_matches = [
                match
                for node in llm_nodes
                if (
                    match := intent_tree_service.match_from_score(
                        str(node.get("node_id") or node.get("id") or ""),
                        float(node.get("score") or 0.0),
                        reason="llm_intent_tree_scoring",
                    )
                )
            ]
            result.guidance_prompt = (
                intent_tree_service.maybe_ambiguity_prompt(result.normalized_question, llm_matches)
                or result.guidance_prompt
            )

        memory = self._memories.get(session_id, SessionMemory())
        expanded_queries = self._build_search_queries(
            result.normalized_question or result.original_question,
            result.rewritten_question,
            result.intent,
            memory,
            result.related_memory,
            sub_questions=result.sub_questions,
            intent_nodes=result.intent_nodes,
        )
        result.search_queries = self._dedupe([*result.search_queries, *expanded_queries])[
            : max(1, config.query_rewrite_max_queries)
        ]
        result.rewrite_strategy = f"{result.rewrite_strategy}+llm_intent_tree"
        result.reasons.append("llm_intent_tree_scoring_applied")
        return result


    def _fuse_confidence(
        self,
        *,
        rule_intent: str,
        rule_confidence: float,
        memory_confidence_floor: float,
        tree_route: str,
        tree_score: float,
        tree_path: str,
        llm_intent: str | None,
        llm_confidence: float | None,
        current_intent: str,
        source: str,
        tree_score_source: str = "rule_intent_tree",
    ) -> tuple[float, dict[str, Any]]:
        rule_confidence = self._clamp_float(rule_confidence, default=0.0)
        memory_confidence_floor = self._clamp_float(memory_confidence_floor, default=0.0)
        tree_score = self._clamp_float(tree_score, default=0.0)
        if llm_confidence is not None:
            llm_confidence = self._clamp_float(llm_confidence, default=0.0)

        weighted_parts: list[tuple[str, float, float]] = [("rule", rule_confidence, 0.35)]
        if memory_confidence_floor > 0:
            weighted_parts.append(("memory", memory_confidence_floor, 0.10))
        if tree_score > 0:
            weighted_parts.append(("intent_tree", tree_score, 0.40))
        if llm_confidence is not None:
            weighted_parts.append(("llm", llm_confidence, 0.25))

        weight_sum = sum(weight for _, _, weight in weighted_parts) or 1.0
        fused = sum(value * weight for _, value, weight in weighted_parts) / weight_sum

        agreement_bonus = 0.0
        conflict_penalty = 0.0
        comparable_tree_route = tree_route or current_intent
        if llm_intent and llm_confidence is not None:
            if llm_intent == rule_intent or llm_intent == comparable_tree_route:
                agreement_bonus = 0.05
            elif llm_confidence >= 0.55:
                conflict_penalty = -0.10

        final_confidence = self._clamp_float(fused + agreement_bonus + conflict_penalty, default=fused)
        return round(final_confidence, 3), {
            "rule": {
                "intent": rule_intent,
                "confidence": round(rule_confidence, 3),
                "source": "keyword_rules",
            },
            "memory": {
                "confidence_floor": round(memory_confidence_floor, 3),
                "applied": memory_confidence_floor > 0,
            },
            "intent_tree": {
                "route": tree_route,
                "score": round(tree_score, 3),
                "path": tree_path,
                "source": tree_score_source,
            },
            "llm": {
                "intent": llm_intent,
                "confidence": round(float(llm_confidence), 3) if llm_confidence is not None else None,
                "applied": llm_confidence is not None,
            },
            "fusion": {
                "source": source,
                "formula": "normalized_weighted_sum(rule*0.35 + memory*0.10 + intent_tree*0.40 + llm*0.25) + agreement_bonus - conflict_penalty",
                "weighted_parts": [
                    {"name": name, "value": round(value, 3), "weight": weight}
                    for name, value, weight in weighted_parts
                ],
                "agreement_bonus": round(agreement_bonus, 3),
                "conflict_penalty": round(conflict_penalty, 3),
                "final_intent": current_intent,
                "final_confidence": round(final_confidence, 3),
            },
        }

    @staticmethod
    def _clamp_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _compact_llm_error(exc: Exception) -> str:
        text = str(exc).strip()
        if not text:
            return exc.__class__.__name__
        lowered = text.lower()
        if "incorrect api key" in lowered or "invalid_api_key" in lowered or "apikey-error" in lowered:
            return "invalid_api_key"
        if "connection error" in lowered:
            return "connection_error"
        if "timeout" in lowered:
            return "timeout"
        return text[:180]

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("LLM query understanding did not return an object")
        return payload

    def update_memory(
        self,
        session_id: str,
        question: str,
        answer: str,
        understanding: QueryUnderstandingResult | None = None,
        user_id: str | None = None,
        persist_long_term: bool = True,
    ) -> SessionMemory:
        """在一轮问答完成后更新增强会话记忆。"""
        memory = self._memories.setdefault(session_id, SessionMemory())
        normalized_question = self._normalize_question(question)
        answer_preview = self._compact_text(answer, 260)
        intent = understanding.intent if understanding else "unknown"
        rewritten = understanding.rewritten_question if understanding else normalized_question

        topic = self._extract_topic(rewritten, intent)
        if topic:
            memory.last_topic = topic
            self._push_topic(memory, topic)

        for fact in self._extract_user_facts(normalized_question):
            if fact not in memory.stable_facts:
                memory.stable_facts.append(fact)
        memory.stable_facts = memory.stable_facts[-12:]
        self._update_user_profile(memory, normalized_question)
        self._merge_entities(memory, self._extract_entities(f"{normalized_question} {answer_preview}"))

        if normalized_question.endswith(("?", "？")):
            memory.focus_questions.append(self._compact_text(normalized_question, 120))
            memory.focus_questions = memory.focus_questions[-8:]

        memory.turn_count += 1
        memory.recent_turns.append(
            {
                "intent": str(intent),
                "topic": topic,
                "question": normalized_question,
                "rewritten_question": rewritten,
                "answer_preview": answer_preview,
                "keywords": self._extract_keywords(f"{normalized_question} {rewritten}"),
            }
        )
        memory.recent_turns = memory.recent_turns[-config.enhanced_memory_recent_turns :]
        memory.summary = self._build_summary(memory)
        if persist_long_term:
            long_term_memory_service.update_from_session(
                user_id=user_id,
                session_id=session_id,
                session_memory=memory.to_dict(),
                question=normalized_question,
                answer_preview=answer_preview,
            )

        logger.debug(
            "增强会话记忆已更新: session={}, turns={}, last_topic={}",
            session_id,
            memory.turn_count,
            memory.last_topic,
        )
        return memory

    def get_memory(self, session_id: str) -> SessionMemory:
        """获取指定会话的增强记忆。"""
        return self._memories.get(session_id, SessionMemory())

    def clear_memory(self, session_id: str) -> None:
        """清理指定会话的增强记忆。"""
        self._memories.pop(session_id, None)

    def _classify_intent(self, question: str) -> tuple[IntentType, float, list[str]]:
        lower_question = question.lower()
        reasons: list[str] = []

        if self._contains_any(lower_question, self.TIME_KEYWORDS):
            reasons.append("命中时间相关关键词")
            return "time_query", 0.88, reasons

        if self._contains_any(lower_question, self.METRICS_KEYWORDS):
            reasons.append("命中 Prometheus/监控告警关键词")
            return "metrics_alerts", 0.86, reasons

        if self._contains_any(lower_question, self.AIOPS_KEYWORDS):
            reasons.append("命中 AIOps/故障诊断关键词")
            return "aiops_diagnosis", 0.84, reasons

        if self._contains_any(lower_question, self.KNOWLEDGE_KEYWORDS):
            reasons.append("命中知识解释/项目实现关键词")
            return "knowledge_query", 0.82, reasons

        if question.strip().endswith("?") or question.strip().endswith("？"):
            reasons.append("疑问句但未命中明确领域，按通用知识问答处理")
            return "knowledge_query", 0.62, reasons

        reasons.append("未命中明确规则，按普通对话处理")
        return "general_chat", 0.55, reasons

    def _rewrite_question(
        self,
        question: str,
        memory: SessionMemory,
        intent: IntentType,
        long_term_topic: str = "",
    ) -> str:
        rewritten = question

        if memory.last_topic and self._looks_like_follow_up(question):
            rewritten = f"结合上文主题“{memory.last_topic}”，{question}"
        elif long_term_topic and self._looks_like_explicit_follow_up(question):
            rewritten = f"结合长期记忆主题“{long_term_topic}”，{question}"

        if intent == "aiops_diagnosis" and "诊断" not in rewritten:
            rewritten = f"针对智能运维故障诊断场景，{rewritten}"
        elif intent == "metrics_alerts" and "Prometheus" not in rewritten:
            rewritten = f"围绕 Prometheus 当前告警和监控指标，{rewritten}"
        elif intent == "knowledge_query" and memory.last_topic and len(question) <= 12:
            rewritten = f"请解释“{memory.last_topic}”中与“{question}”相关的概念和实现细节"

        return rewritten

    def _split_sub_questions(self, original_question: str, rewritten_question: str) -> list[str]:
        """Ragent-style deterministic question split with conservative rules."""
        source = original_question.strip()
        if not source:
            return [rewritten_question]

        has_explicit_split = (
            source.count("？") + source.count("?") >= 2
            or any(mark in source for mark in ("；", ";", "\n"))
            or bool(re.search(r"(?:^|\s)[1-9][).、]", source))
        )
        if not has_explicit_split:
            return [rewritten_question]

        parts = [
            item.strip(" ，。？！?;；\t")
            for item in re.split(r"[?？。；;\n]+|(?:^|\s)[1-9][).、]", source)
            if item.strip(" ，。？！?;；\t")
        ]
        if len(parts) <= 1:
            return [rewritten_question]

        expanded: list[str] = []
        anchor = self._infer_split_anchor(parts)
        for part in parts:
            is_follow_up_part = part.startswith(("那", "如果", "那如果", "这个", "这种", "它", "继续"))
            if anchor and part != anchor and (
                is_follow_up_part or (len(part) <= 12 and not self._has_independent_domain_signal(part))
            ):
                expanded.append(f"{anchor} {part}")
            else:
                expanded.append(part)
        return self._dedupe(expanded)[: max(1, config.query_rewrite_max_queries)]

    def _infer_split_anchor(self, parts: list[str]) -> str:
        for part in parts:
            if self._has_independent_domain_signal(part):
                return part
        return ""

    def _build_search_queries(
        self,
        original_question: str,
        rewritten_question: str,
        intent: IntentType,
        memory: SessionMemory,
        related_turns: list[dict[str, Any]] | None = None,
        *,
        sub_questions: list[str] | None = None,
        intent_nodes: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        queries = [rewritten_question]
        if rewritten_question != original_question:
            queries.append(original_question)
        for sub_question in sub_questions or []:
            if sub_question and sub_question != rewritten_question:
                queries.append(sub_question)
        for node in (intent_nodes or [])[:3]:
            node_name = str(node.get("name") or "").strip()
            patterns = [
                str(item).strip()
                for item in node.get("source_patterns", [])
                if str(item).strip()
            ][:3]
            if node_name:
                queries.append(" ".join([node_name, *patterns, original_question]).strip())

        combined_question = f"{original_question} {rewritten_question}"
        lower_question = combined_question.lower()
        if "cpu" in lower_question or "负载" in combined_question or "load" in lower_question:
            queries.append("CPU 使用率过高 告警 排查 处理方案")
        if "java" in lower_question and ("cpu" in lower_question or "负载" in combined_question):
            queries.append("Java 服务 CPU 高 jstack jcmd 线程栈 火焰图 排查")
        if "内存" in combined_question or "memory" in lower_question or "oom" in lower_question or "gc" in lower_question:
            queries.append("内存使用率过高 内存泄漏 排查 处理方案")
        if "磁盘" in combined_question or "disk" in lower_question:
            queries.append("磁盘使用率过高 磁盘空间不足 排查 处理方案")
        if "响应慢" in combined_question or "慢" in combined_question or "超时" in combined_question or "流量突增" in combined_question:
            queries.append("服务响应慢 请求超时 性能问题 排查")
        if "服务不可用" in combined_question or "宕机" in combined_question or "打不开" in combined_question or "挂了" in combined_question or "回滚" in combined_question or "下游" in combined_question:
            queries.append("服务不可用 服务宕机 故障排查 处理方案")
        if intent == "knowledge_query" and memory.last_topic:
            queries.append(f"{memory.last_topic} {original_question}")
        if related_turns and self._looks_like_follow_up(original_question):
            for turn in related_turns[:2]:
                related_topic = str(turn.get("topic") or memory.last_topic or "").strip()
                related_question = str(turn.get("question") or "").strip()
                if related_topic or related_question:
                    queries.append(f"{related_topic} {related_question} {original_question}".strip())

        return self._dedupe(queries)[: max(1, config.query_rewrite_max_queries)]

    def _build_route_hint(self, intent: IntentType) -> str:
        hints = {
            "aiops_diagnosis": "优先考虑 Prometheus 告警、MCP 监控/日志工具和知识库处理方案。",
            "knowledge_query": "优先考虑知识库检索工具 retrieve_knowledge，再结合模型解释。",
            "metrics_alerts": "优先考虑 Prometheus 告警查询工具 query_prometheus_alerts。",
            "time_query": "优先考虑时间工具 get_current_time。",
            "general_chat": "可直接回答；如涉及项目/文档内容再调用知识库。",
            "unknown": "无法确定意图时，先澄清或使用通用工具判断。",
        }
        return hints[intent]

    def _format_memory_context(
        self,
        memory: SessionMemory,
        current_question: str | None = None,
        related_turns: list[dict[str, Any]] | None = None,
        long_term_context: str = "",
        intent: str = "",
        include_topic_context: bool | None = None,
    ) -> str:
        parts: list[str] = []
        related = related_turns if related_turns is not None else self._retrieve_relevant_turns(memory, current_question or "")
        should_include_topic_context = (
            include_topic_context
            if include_topic_context is not None
            else bool(current_question and self._looks_like_follow_up(current_question)) or bool(related)
        )
        if long_term_context:
            parts.append("长期用户记忆:\n" + long_term_context)
        if should_include_topic_context and memory.summary and not related:
            parts.append(f"会话摘要: {memory.summary}")
        if should_include_topic_context and memory.last_topic:
            parts.append(f"最近主题: {memory.last_topic}")
        if should_include_topic_context and intent == "knowledge_query" and memory.topic_stack:
            parts.append("主题轨迹: " + " -> ".join(memory.topic_stack[-5:]))
        if memory.user_profile:
            profile_items = self._format_profile_items(
                memory.user_profile,
                include_project_preferences=intent == "knowledge_query",
            )
            if profile_items:
                parts.append("用户画像: " + "；".join(profile_items[:6]))
        if memory.stable_facts:
            stable_facts = self._filter_memory_facts(
                memory.stable_facts,
                include_project_preferences=intent == "knowledge_query",
            )
            if stable_facts:
                parts.append("稳定用户信息: " + "；".join(stable_facts[-6:]))
        entity_lines = []
        if should_include_topic_context:
            for entity_type, values in memory.entities.items():
                if intent != "knowledge_query" and entity_type not in {"fault", "metric", "service"}:
                    continue
                if values:
                    entity_lines.append(f"{entity_type}: {', '.join(values[-6:])}")
        if entity_lines:
            parts.append("已识别实体: " + "；".join(entity_lines[:5]))
        if should_include_topic_context and intent == "knowledge_query" and memory.focus_questions:
            parts.append("近期关注问题: " + "；".join(memory.focus_questions[-3:]))
        if related:
            lines = []
            for turn in related[: config.enhanced_memory_relevant_turns]:
                lines.append(
                    f"- 主题: {turn.get('topic', '')}\n"
                    f"  用户: {turn.get('question', '')}\n"
                    f"  回答摘要: {turn.get('answer_preview', '')}"
                )
            parts.append("相关历史对话:\n" + "\n".join(lines))
        if should_include_topic_context and memory.recent_turns:
            recent = []
            recent_source = related[:2] if related else memory.recent_turns[-1:]
            for turn in recent_source:
                recent.append(
                    f"- 用户: {turn.get('question', '')}\n"
                    f"  改写: {turn.get('rewritten_question', '')}\n"
                    f"  回答摘要: {turn.get('answer_preview', '')}"
                )
            parts.append("最近对话:\n" + "\n".join(recent))

        return "\n".join(parts)

    def _build_summary(self, memory: SessionMemory) -> str:
        recent_topics = []
        for turn in memory.recent_turns[-config.enhanced_memory_recent_turns :]:
            topic = self._extract_topic(turn.get("rewritten_question", ""), turn.get("intent", "unknown"))
            if topic and topic not in recent_topics:
                recent_topics.append(topic)

        summary_parts = []
        if memory.last_topic:
            summary_parts.append(f"当前主要讨论主题是“{memory.last_topic}”。")
        if recent_topics:
            summary_parts.append("近期涉及: " + "、".join(recent_topics[-5:]) + "。")
        if memory.topic_stack:
            summary_parts.append("主题轨迹: " + " -> ".join(memory.topic_stack[-4:]) + "。")
        if memory.stable_facts:
            summary_parts.append("用户背景/偏好: " + "；".join(memory.stable_facts[-5:]) + "。")
        if memory.focus_questions:
            summary_parts.append("近期关注: " + "；".join(memory.focus_questions[-3:]) + "。")
        if memory.entities:
            entity_parts = []
            for entity_type, values in memory.entities.items():
                if values:
                    entity_parts.append(f"{entity_type}={','.join(values[-3:])}")
            if entity_parts:
                summary_parts.append("关键实体: " + "；".join(entity_parts[:4]) + "。")
        if memory.recent_turns:
            last_turn = memory.recent_turns[-1]
            summary_parts.append(
                "上一轮用户询问: "
                + self._compact_text(last_turn.get("question", ""), 80)
                + "；回答要点: "
                + self._compact_text(last_turn.get("answer_preview", ""), 120)
                + "。"
            )

        summary = "".join(summary_parts)
        return self._compact_text(summary, config.enhanced_memory_summary_max_chars)

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
                profile_items.append(f"{key}={','.join(str(item) for item in value[-4:])}")
            else:
                profile_items.append(f"{key}={value}")
        return profile_items

    def _filter_memory_facts(
        self,
        facts: list[str],
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
        result = []
        for fact in facts:
            text = str(fact).strip()
            if not text:
                continue
            if any(marker in text for marker in project_markers):
                continue
            result.append(text)
        return result

    def _extract_topic(self, text: str, intent: str) -> str:
        cleaned = self._normalize_question(text)
        prefixes = (
            "结合上文主题",
            "针对智能运维故障诊断场景",
            "围绕 Prometheus 当前告警和监控指标",
            "请解释",
        )
        for prefix in prefixes:
            cleaned = cleaned.replace(prefix, "")
        cleaned = cleaned.strip(" ，。？！?\"“”")

        known_topics = (
            "RAG",
            "MCP",
            "Milvus",
            "Prometheus",
            "LangGraph",
            "LangChain",
            "Agent",
            "Embedding",
            "Collection",
            "CORS",
            "lifespan",
            "AIOps",
            "Redis",
            "RDB",
            "AOF",
        )
        lower_cleaned = cleaned.lower()
        if "baseline" in lower_cleaned or "评测" in cleaned or "量化" in cleaned:
            return "RAG 量化评测"
        if "记忆" in cleaned or "memory" in lower_cleaned:
            return "增强会话记忆"
        if "面试" in cleaned and ("项目" in cleaned or "亮点" in cleaned or "简历" in cleaned):
            return "项目面试表达"
        if "代码小白" in cleaned or "通俗" in cleaned:
            return "学习解释偏好"
        for topic in known_topics:
            if topic.lower() in lower_cleaned:
                return topic

        if intent == "aiops_diagnosis":
            if "cpu" in lower_cleaned or "负载" in cleaned or "load" in lower_cleaned:
                return "CPU 高负载排查"
            if "内存" in cleaned or "oom" in lower_cleaned or "gc" in lower_cleaned:
                return "内存故障排查"
            if "磁盘" in cleaned or "inode" in lower_cleaned or "空间" in cleaned:
                return "磁盘空间故障排查"
            if "响应慢" in cleaned or "超时" in cleaned or "慢" in cleaned:
                return "服务响应慢排查"
            if "不可用" in cleaned or "打不开" in cleaned or "挂了" in cleaned or "5xx" in lower_cleaned:
                return "服务不可用排查"
            return "AIOps 智能运维诊断"
        if intent == "metrics_alerts":
            return "Prometheus 告警与监控"
        if intent == "knowledge_query":
            return self._compact_text(cleaned, 32)
        return self._compact_text(cleaned, 24)

    def _latest_long_term_topic(self, memory: dict[str, Any]) -> str:
        topic_stack = memory.get("topic_stack") or []
        if topic_stack:
            return str(topic_stack[-1])
        recent_sessions = memory.get("recent_sessions") or []
        if recent_sessions:
            return str(recent_sessions[-1].get("last_topic", ""))
        return ""

    def _should_use_long_term_topic(
        self,
        memory: SessionMemory,
        question: str,
        intent: IntentType,
        long_term_topic: str,
    ) -> bool:
        """Only use old cross-session topics when they are clearly relevant."""
        if not config.enable_long_term_user_memory:
            return False
        if memory.last_topic or not long_term_topic:
            return False
        if not self._looks_like_explicit_follow_up(question):
            return False
        return self._topic_matches_question(long_term_topic, question, intent)

    def _topic_matches_question(self, topic: str, question: str, intent: IntentType) -> bool:
        topic_lower = topic.lower()
        question_lower = question.lower()
        if topic_lower and topic_lower in question_lower:
            return True

        topic_keywords = set(self._extract_keywords(topic))
        question_keywords = set(self._extract_keywords(question))
        if topic_keywords & question_keywords:
            return True

        aiops_topic_markers = ("aiops", "运维", "故障", "诊断", "cpu", "内存", "磁盘", "服务")
        if intent == "aiops_diagnosis" and any(marker in topic_lower for marker in aiops_topic_markers):
            return True
        return False

    def _push_topic(self, memory: SessionMemory, topic: str) -> None:
        topic = topic.strip()
        if not topic:
            return
        memory.topic_stack = [item for item in memory.topic_stack if item != topic]
        memory.topic_stack.append(topic)
        memory.topic_stack = memory.topic_stack[-config.enhanced_memory_topic_stack_size :]

    def _update_user_profile(self, memory: SessionMemory, question: str) -> None:
        profile = memory.user_profile

        def add_list(key: str, value: str) -> None:
            values = profile.setdefault(key, [])
            if not isinstance(values, list):
                values = [str(values)]
            if value not in values:
                values.append(value)
            profile[key] = values[-6:]

        if "代码小白" in question or "不懂代码" in question or "完全不熟" in question:
            profile["explanation_level"] = "beginner_friendly"
            add_list("answer_style", "通俗解释")
        if "详细" in question or "越全面越好" in question:
            add_list("answer_style", "详细拆解")
        if "面试" in question:
            add_list("goals", "面试准备")
        if "简历" in question:
            add_list("goals", "简历包装")
        if "技术力" in question or "亮眼" in question or "不够" in question:
            add_list("goals", "提升项目技术亮点")
        if "量化" in question or "评测" in question or "baseline" in question.lower():
            add_list("goals", "量化评测")
        if "页面" in question or "界面" in question or "前端" in question or "展示" in question:
            add_list("preferences", "前端可视化展示")
        if "不想通过powershell" in question.lower() or "回答结果查看" in question:
            add_list("preferences", "优先通过页面查看结果")

    def _merge_entities(self, memory: SessionMemory, entities: dict[str, list[str]]) -> None:
        limit = max(1, config.enhanced_memory_entity_limit)
        for entity_type, values in entities.items():
            if not values:
                continue
            merged = list(memory.entities.get(entity_type, []))
            for value in values:
                if value and value not in merged:
                    merged.append(value)
            memory.entities[entity_type] = merged[-limit:]

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        lower_text = text.lower()
        entities: dict[str, list[str]] = {
            "tech": [],
            "fault": [],
            "file": [],
            "metric": [],
        }

        tech_terms = (
            "RAG",
            "MCP",
            "Prometheus",
            "Milvus",
            "FastAPI",
            "Redis",
            "LangGraph",
            "LangChain",
            "DashScope",
            "Qwen",
            "Docker",
            "CLS",
            "Baseline",
            "Rerank",
        )
        for term in tech_terms:
            if term.lower() in lower_text:
                entities["tech"].append(term)

        fault_terms = (
            "CPU",
            "load",
            "内存",
            "OOM",
            "GC",
            "heap dump",
            "磁盘",
            "inode",
            "响应慢",
            "超时",
            "服务不可用",
            "5xx",
            "503",
            "回滚",
        )
        for term in fault_terms:
            if term.lower() in lower_text:
                entities["fault"].append(term)

        for file_name in re.findall(r"[\w\u4e00-\u9fff.-]+\.(?:md|txt|py|json|ps1)", text, flags=re.IGNORECASE):
            entities["file"].append(file_name)

        for metric in re.findall(r"\b(?:hit@?\d+|mrr|p99|qps|rt|cpu|rss|gc|oom|5xx)\b", lower_text):
            entities["metric"].append(metric.upper() if metric in {"cpu", "rss", "gc", "oom"} else metric)

        return {key: self._dedupe(values) for key, values in entities.items() if values}

    def _extract_keywords(self, text: str) -> list[str]:
        lower_text = text.lower()
        keywords: list[str] = []
        domain_terms = (
            "cpu",
            "负载",
            "load",
            "内存",
            "memory",
            "oom",
            "gc",
            "dump",
            "磁盘",
            "inode",
            "响应慢",
            "超时",
            "服务不可用",
            "回滚",
            "redis",
            "rdb",
            "aof",
            "rag",
            "mcp",
            "baseline",
            "评测",
            "重排",
            "记忆",
        )
        for term in domain_terms:
            if term in lower_text:
                keywords.append(term)

        stopwords = {"这个", "那个", "怎么", "什么", "是否", "应该", "可以", "如果", "然后", "就是"}
        for token in re.findall(r"[a-zA-Z0-9_@]+|[\u4e00-\u9fff]{2,}", lower_text):
            if token in stopwords or len(token) > 24:
                continue
            keywords.append(token)
        return self._dedupe(keywords)[:16]

    def _retrieve_relevant_turns(self, memory: SessionMemory, question: str) -> list[dict[str, Any]]:
        if not memory.recent_turns:
            return []

        question_keywords = set(self._extract_keywords(question))
        question_lower = question.lower()
        is_follow_up = self._looks_like_follow_up(question)
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for idx, turn in enumerate(memory.recent_turns):
            turn_text = " ".join(
                str(turn.get(key, ""))
                for key in ("topic", "question", "rewritten_question", "answer_preview")
            )
            turn_keywords = set(turn.get("keywords") or self._extract_keywords(turn_text))
            overlap = question_keywords & turn_keywords
            score = float(len(overlap) * 2)

            topic = str(turn.get("topic") or "")
            if topic and (topic in question or any(keyword in topic.lower() for keyword in question_keywords)):
                score += 3.0

            for values in memory.entities.values():
                for value in values:
                    value_lower = value.lower()
                    if value_lower and value_lower in question_lower and value_lower in turn_text.lower():
                        score += 1.5

            if is_follow_up and idx == len(memory.recent_turns) - 1:
                score += 1.0

            if score <= 0:
                continue
            scored.append((score, idx, turn))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        related = []
        for score, _, turn in scored[: config.enhanced_memory_relevant_turns]:
            related.append(
                {
                    "score": round(score, 2),
                    "intent": turn.get("intent", ""),
                    "topic": turn.get("topic", ""),
                    "question": turn.get("question", ""),
                    "rewritten_question": turn.get("rewritten_question", ""),
                    "answer_preview": turn.get("answer_preview", ""),
                    "keywords": turn.get("keywords", []),
                }
            )
        return related

    def _extract_user_facts(self, question: str) -> list[str]:
        facts: list[str] = []
        if "代码小白" in question or "不懂代码" in question or "完全不熟" in question:
            facts.append("用户偏代码小白，需要用通俗方式解释")
        if "面试" in question:
            facts.append("用户关注面试表达和项目包装")
        if "详细" in question or "越全面越好" in question:
            facts.append("用户偏好详细全面的解释")
        if "技术力" in question or "亮眼" in question:
            facts.append("用户希望项目体现更强技术深度")
        if "量化" in question or "评测" in question or "baseline" in question.lower():
            facts.append("用户关注可量化效果和 Baseline 对比")
        if "页面" in question or "界面" in question or "前端" in question or "展示" in question:
            facts.append("用户希望功能能在页面直观看到")
        if "重复" in question or "撞项目" in question:
            facts.append("用户希望降低项目同质化")
        return facts

    def _looks_like_follow_up(self, question: str) -> bool:
        if self._contains_any(question, self.FOLLOW_UP_MARKERS):
            return True
        if len(question) <= 16:
            return not self._has_independent_domain_signal(question)
        return False

    def _looks_like_explicit_follow_up(self, question: str) -> bool:
        return self._contains_any(question, self.FOLLOW_UP_MARKERS)

    def _has_independent_domain_signal(self, question: str) -> bool:
        lower_question = question.lower()
        domain_keywords = (
            *self.AIOPS_KEYWORDS,
            *self.METRICS_KEYWORDS,
            *self.KNOWLEDGE_KEYWORDS,
            "cpu",
            "java",
            "jvm",
            "jstack",
            "jcmd",
            "redis",
            "rag",
            "mcp",
            "prometheus",
            "数据库",
            "日志",
        )
        return self._contains_any(lower_question, domain_keywords)

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.lower() in text.lower() for keyword in keywords)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.strip().split())

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        compacted = " ".join(str(text).strip().split())
        if len(compacted) <= max_chars:
            return compacted
        return compacted[: max_chars - 3] + "..."

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


query_understanding_service = QueryUnderstandingService()
