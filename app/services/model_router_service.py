"""Lightweight model routing and failover for ChatQwen models."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config


@dataclass
class ModelHealth:
    model: str
    failures: int = 0
    open_until: float = 0.0
    last_error: str = ""

    @property
    def available(self) -> bool:
        return self.open_until <= time.time()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["available"] = self.available
        return data


class ModelRouterService:
    """Select a healthy model and track simple circuit-breaker state."""

    def __init__(self) -> None:
        self._health: dict[str, ModelHealth] = {}

    def candidate_models(self, preferred: str | None = None) -> list[str]:
        raw_candidates = [
            item.strip()
            for item in config.llm_candidate_models.split(",")
            if item.strip()
        ]
        if preferred and preferred not in raw_candidates:
            raw_candidates.insert(0, preferred)
        if not raw_candidates:
            raw_candidates = [config.rag_model]

        seen: set[str] = set()
        candidates: list[str] = []
        for model in raw_candidates:
            if model not in seen:
                seen.add(model)
                candidates.append(model)
        return candidates

    def available_models(self, preferred: str | None = None) -> list[str]:
        if not config.enable_model_routing:
            return [preferred or config.rag_model]

        candidates = self.candidate_models(preferred)
        available = [
            model
            for model in candidates
            if self._health.setdefault(model, ModelHealth(model=model)).available
        ]
        return available or candidates[:1]

    def create_chat_model(self, model: str, streaming: bool) -> ChatQwen:
        logger.info("创建 ChatQwen 模型实例: model={}, streaming={}", model, streaming)
        return ChatQwen(
            model=model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0.7,
            streaming=streaming,
        )

    def mark_success(self, model: str) -> None:
        health = self._health.setdefault(model, ModelHealth(model=model))
        health.failures = 0
        health.open_until = 0.0
        health.last_error = ""

    def mark_failure(self, model: str, error: Exception | str) -> None:
        health = self._health.setdefault(model, ModelHealth(model=model))
        health.failures += 1
        health.last_error = str(error)
        if health.failures >= max(1, config.llm_failure_threshold):
            health.open_until = time.time() + max(1, config.llm_cooldown_seconds)
            logger.warning(
                "模型临时熔断: model={}, failures={}, cooldown={}s",
                model,
                health.failures,
                config.llm_cooldown_seconds,
            )

    def health_snapshot(self) -> dict[str, Any]:
        for model in self.candidate_models(config.rag_model):
            self._health.setdefault(model, ModelHealth(model=model))
        return {model: health.to_dict() for model, health in self._health.items()}


model_router_service = ModelRouterService()
