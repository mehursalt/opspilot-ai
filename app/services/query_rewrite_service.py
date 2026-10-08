"""LLM structured query rewrite service.

This mirrors the Ragent-style rewrite stage: ask a small model to return a
strict JSON object, then let QueryUnderstandingService merge it with the
deterministic baseline. The service is optional and safe to skip when the API
key is missing or the model call fails.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import config
from app.services.model_router_service import model_router_service


class QueryRewriteService:
    """Generate structured query-understanding JSON with an LLM."""

    async def rewrite(self, question: str, baseline: dict[str, Any]) -> dict[str, Any]:
        if not config.dashscope_api_key:
            raise RuntimeError("dashscope_api_key is empty")

        llm = model_router_service.create_chat_model(
            config.llm_query_understanding_model,
            streaming=False,
        )
        prompt = self._build_prompt(question, baseline)
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", response)
        payload = self._parse_json_object(str(content))
        return self._normalize_payload(payload)

    def _build_prompt(self, question: str, baseline: dict[str, Any]) -> str:
        return f"""
你是一个智能运维 RAG Agent 的 QueryRewriteService。你的任务不是回答问题，而是把用户问题改写成适合检索、路由和工具调用的结构化 JSON。

请严格遵守：
1. 只输出 JSON 对象，不要输出 Markdown、解释或多余文本。
2. rewritten_question 必须补全省略指代，但不能改变用户真实意图。
3. 如果用户问题包含追问、并列问题、多个故障对象，请拆成 sub_questions。
4. search_queries 要面向知识库检索，包含同义表达、排障术语、工具关键词。
5. missing_slots 用来列出还缺的诊断槽位，例如 region、service、time_range、log_topic。
6. confidence 是你对本次结构化理解的置信度，范围 0 到 1。

可选 intent:
- aiops_diagnosis: CPU、内存、磁盘、响应慢、服务不可用等排障问题
- knowledge_query: 项目知识、Redis、RAG、MCP、代码实现、面试讲解
- metrics_alerts: Prometheus、告警、监控指标查询
- time_query: 时间日期类问题
- general_chat: 普通聊天
- unknown: 无法判断

用户原始问题:
{question}

规则基线结果:
{json.dumps(baseline, ensure_ascii=False)}

请输出 JSON，schema 如下：
{{
  "intent": "aiops_diagnosis",
  "confidence": 0.8,
  "rewritten_question": "...",
  "should_split": true,
  "sub_questions": ["..."],
  "search_queries": ["...", "..."],
  "missing_slots": ["..."],
  "rewrite_reason": "..."
}}
""".strip()

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        valid_intents = {
            "aiops_diagnosis",
            "knowledge_query",
            "metrics_alerts",
            "time_query",
            "general_chat",
            "unknown",
        }
        intent = str(payload.get("intent") or "unknown").strip()
        if intent not in valid_intents:
            intent = "unknown"

        confidence = self._float_between(payload.get("confidence"), default=0.0)
        rewritten_question = str(payload.get("rewritten_question") or "").strip()
        sub_questions = self._list_of_str(payload.get("sub_questions"))[:4]
        search_queries = self._list_of_str(payload.get("search_queries"))[:6]
        missing_slots = self._list_of_str(payload.get("missing_slots"))[:8]
        rewrite_reason = str(payload.get("rewrite_reason") or "").strip()

        return {
            "intent": intent,
            "confidence": confidence,
            "rewritten_question": rewritten_question,
            "should_split": bool(payload.get("should_split", len(sub_questions) > 1)),
            "sub_questions": sub_questions,
            "search_queries": search_queries,
            "missing_slots": missing_slots,
            "reasons": [rewrite_reason] if rewrite_reason else [],
            "rewrite_reason": rewrite_reason,
        }

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
            raise ValueError("query rewrite LLM did not return a JSON object")
        return payload

    @staticmethod
    def _list_of_str(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _float_between(value: Any, *, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default


query_rewrite_service = QueryRewriteService()
