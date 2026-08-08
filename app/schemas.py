from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "coding",
    "writing",
    "summarisation",
    "translation",
    "reasoning",
    "general",
]

Priority = Literal["cheap", "balanced", "quality"]


class RouteRequest(BaseModel):
    query: str = Field(min_length=1)
    system: str | None = None
    priority: Priority | None = None
    max_tokens: int = Field(default=1024, ge=1, le=32000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    model_hint: str | None = Field(default=None, description="Force a model id, skips routing")
    stream: bool = False
    cache: bool = True


class Classification(BaseModel):
    allowed: bool = True
    reason: str | None = None
    intent: Intent = "general"
    confidence: float = 0.5
    complexity: Literal["low", "medium", "high"] = "medium"
    needs_tools: bool = False
    source: Literal["heuristic", "llm", "cache"] = "heuristic"


class Decision(BaseModel):
    model: str
    provider: str
    intent: Intent
    priority: Priority
    score: float
    est_cost_usd: float
    candidates: list[dict]
    reason: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class RouteResponse(BaseModel):
    id: str
    output: str
    decision: Decision
    classification: Classification
    usage: Usage
    latency_ms: int
    cached: Literal[False, "semantic", "idempotent"] = False


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
