"""
Intent detection interfaces and LLM-based detector for VerbatimRAG.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from verbatim_rag.llm_client import LLMClient


@dataclass
class IntentDecision:
    intent: str
    route: str
    answer: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None


class IntentDetector(ABC):
    """Abstract base class for intent detection."""

    @abstractmethod
    def detect(self, question: str) -> IntentDecision:
        raise NotImplementedError

    async def detect_async(self, question: str) -> IntentDecision:
        return await asyncio.to_thread(self.detect, question)


DEFAULT_INTENT_PROMPT = """You are an intent router for a RAG system.
Return JSON only with the following schema:
{"intent":"string","route":"continue|predefined|skip","answer":"string|null","confidence":0.0,"reason":"string"}
If unsure, use route "continue" and set a low confidence.
"""


class LLMIntentDetector(IntentDetector):
    """
    LLM-based intent detector with configurable prompt, examples, and routes.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt: str | None = None,
        examples: Optional[list[dict[str, Any]]] = None,
        routes: Optional[dict[str, dict[str, Any]]] = None,
        min_confidence: float = 0.0,
        fallback_route: str = "continue",
        fallback_answer: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.prompt = prompt or DEFAULT_INTENT_PROMPT
        self.examples = examples or []
        self.routes = routes or {}
        self.min_confidence = min_confidence
        self.fallback_route = fallback_route
        self.fallback_answer = fallback_answer

    @classmethod
    def from_config(
        cls, llm_client: LLMClient, config: dict[str, Any]
    ) -> "LLMIntentDetector":
        return cls(
            llm_client=llm_client,
            prompt=config.get("prompt"),
            examples=config.get("examples"),
            routes=config.get("routes"),
            min_confidence=config.get("confidence", {}).get("min", 0.0),
            fallback_route=config.get("fallback", {}).get("route", "continue"),
            fallback_answer=config.get("fallback", {}).get("answer"),
        )

    def detect(self, question: str) -> IntentDecision:
        prompt = self._build_prompt(question)
        response = self.llm_client.complete(prompt, json_mode=True)
        return self._parse_response(response)

    async def detect_async(self, question: str) -> IntentDecision:
        prompt = self._build_prompt(question)
        response = await self.llm_client.complete_async(prompt, json_mode=True)
        return self._parse_response(response)

    def _build_prompt(self, question: str) -> str:
        lines = [self.prompt.strip(), "", f"Question: {question}"]
        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for ex in self.examples:
                q = ex.get("question", "")
                example = {
                    "intent": ex.get("intent", ""),
                    "route": ex.get("route", "continue"),
                    "answer": ex.get("answer"),
                    "confidence": ex.get("confidence", 0.8),
                    "reason": ex.get("reason", ""),
                }
                lines.append(f'Q: "{q}"')
                lines.append(f"A: {json.dumps(example, ensure_ascii=True)}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> IntentDecision:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return self._fallback_decision("invalid_json")

        intent = payload.get("intent", "unknown")
        route = payload.get("route") or self.fallback_route
        answer = payload.get("answer", None)
        confidence = payload.get("confidence", None)
        reason = payload.get("reason", None)

        if isinstance(confidence, (int, float)) and confidence < self.min_confidence:
            return self._fallback_decision("low_confidence")

        if intent in self.routes:
            override = self.routes[intent]
            route = override.get("route", route)
            answer = override.get("answer", answer)

        if route not in {"continue", "predefined", "skip"}:
            return self._fallback_decision("invalid_route")

        return IntentDecision(
            intent=intent,
            route=route,
            answer=answer,
            confidence=confidence,
            reason=reason,
        )

    def _fallback_decision(self, reason: str) -> IntentDecision:
        return IntentDecision(
            intent="fallback",
            route=self.fallback_route,
            answer=self.fallback_answer,
            confidence=0.0,
            reason=reason,
        )
