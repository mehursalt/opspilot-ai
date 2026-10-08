"""Intent guidance service.

When query understanding confidence is low, this service builds a short
clarification message instead of letting the Agent guess blindly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.config import config
from app.services.query_understanding_service import QueryUnderstandingResult


@dataclass
class GuidanceDecision:
    """Decision returned after ambiguity / confidence checking."""

    should_guide: bool
    message: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntentGuidanceService:
    """Judge whether the current question should be clarified first."""

    def evaluate(self, understanding: QueryUnderstandingResult) -> GuidanceDecision:
        if understanding.guidance_prompt:
            return GuidanceDecision(
                should_guide=True,
                message=understanding.guidance_prompt,
                reasons=[
                    "intent_tree_ambiguity",
                    f"intent={understanding.intent}",
                    f"confidence={understanding.confidence:.2f}",
                ],
            )

        if understanding.confidence >= config.intent_guidance_min_confidence:
            return GuidanceDecision(should_guide=False)

        if understanding.intent not in ("general_chat", "unknown"):
            return GuidanceDecision(should_guide=False)

        message = (
            "我还不能确定你想让我走哪条处理链路。你可以补充一下："
            "是要查项目知识/RAG，做 AIOps 故障排查，查询 Prometheus 告警，"
            "还是只是普通解释？例如可以问“CPU 过高怎么排查”或“什么是 MCP”。"
        )
        return GuidanceDecision(
            should_guide=True,
            message=message,
            reasons=[
                f"intent={understanding.intent}",
                f"confidence={understanding.confidence:.2f} < {config.intent_guidance_min_confidence:.2f}",
            ],
        )


intent_guidance_service = IntentGuidanceService()
