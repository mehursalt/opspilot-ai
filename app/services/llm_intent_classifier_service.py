"""LLM-assisted intent tree classifier.

The deterministic intent tree is still the fallback. This service asks a small
model to score configured tree nodes only, so the model cannot invent unknown
routes. It is inspired by Ragent's leaf-node scoring stage.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import config
from app.services.intent_tree_service import IntentNodeMatch, intent_tree_service
from app.services.model_router_service import model_router_service


class LLMIntentClassifierService:
    """Score configured intent-tree nodes with an LLM."""

    async def classify(
        self,
        *,
        question: str,
        rewritten_question: str,
        sub_questions: list[str],
        baseline_matches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not config.dashscope_api_key:
            raise RuntimeError("dashscope_api_key is empty")

        catalog = intent_tree_service.node_catalog()
        if not catalog:
            return {"matches": [], "guidance_prompt": "", "raw": {}}

        llm = model_router_service.create_chat_model(
            config.llm_intent_tree_model,
            streaming=False,
        )
        prompt = self._build_prompt(
            question=question,
            rewritten_question=rewritten_question,
            sub_questions=sub_questions,
            baseline_matches=baseline_matches,
            catalog=catalog,
        )
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", response)
        payload = self._parse_json_object(str(content))
        matches = self._matches_from_payload(payload)
        return {
            "matches": [match.to_dict() for match in matches],
            "guidance_prompt": str(payload.get("guidance_prompt") or "").strip(),
            "need_guidance": bool(payload.get("need_guidance", False)),
            "raw": payload,
        }

    def _build_prompt(
        self,
        *,
        question: str,
        rewritten_question: str,
        sub_questions: list[str],
        baseline_matches: list[dict[str, Any]],
        catalog: list[dict[str, Any]],
    ) -> str:
        return f"""
你是智能运维 RAG Agent 的 IntentResolver。请从给定意图树节点中选择最匹配用户问题的候选节点并打分。

严格要求：
1. 只输出 JSON 对象，不要输出 Markdown。
2. 只能选择 catalog 中存在的 id，禁止创造新 id。
3. score 范围 0 到 1，0.55 以上表示可用于定向检索。
4. 如果 Top1 和 Top2 很接近、用户问题缺少关键对象，请设置 need_guidance=true 并生成 guidance_prompt。
5. reason 要简短说明为什么命中该节点。

用户原始问题:
{question}

重写问题:
{rewritten_question}

子问题:
{json.dumps(sub_questions, ensure_ascii=False)}

规则候选:
{json.dumps(baseline_matches, ensure_ascii=False)}

意图树目录:
{json.dumps(catalog, ensure_ascii=False)}

请输出 JSON：
{{
  "candidates": [
    {{"id": "ops.cpu", "score": 0.92, "reason": "用户明确询问 CPU 过高排查"}}
  ],
  "need_guidance": false,
  "guidance_prompt": ""
}}
""".strip()

    def _matches_from_payload(self, payload: dict[str, Any]) -> list[IntentNodeMatch]:
        raw_candidates = payload.get("candidates") or []
        if not isinstance(raw_candidates, list):
            return []

        matches: list[IntentNodeMatch] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id") or item.get("node_id") or "").strip()
            score = self._float_between(item.get("score"), default=0.0)
            if score < config.intent_tree_min_score:
                continue
            reason = str(item.get("reason") or "llm_intent_score").strip()
            match = intent_tree_service.match_from_score(
                node_id,
                score,
                reason=reason,
                matched_terms=[f"llm:{reason}"] if reason else ["llm_intent_score"],
            )
            if match:
                matches.append(match)

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: max(1, config.intent_tree_max_candidates)]

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
            raise ValueError("intent classifier LLM did not return a JSON object")
        return payload

    @staticmethod
    def _float_between(value: Any, *, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default


llm_intent_classifier_service = LLMIntentClassifierService()
