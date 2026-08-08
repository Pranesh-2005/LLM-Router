"""Model catalog: prices + context windows from LiteLLM's public price sheet,
quality/speed scores from data/quality.json (refreshed by scripts/update_catalog.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import DATA_DIR

log = logging.getLogger(__name__)

PRICES_FILE = DATA_DIR / "model_prices.json"
QUALITY_FILE = DATA_DIR / "quality.json"


@dataclass
class ModelInfo:
    id: str
    provider: str
    input_cost: float = 0.0        # USD per token
    output_cost: float = 0.0
    context: int = 8192
    max_output: int = 4096
    supports_tools: bool = False
    quality: float = 0.6           # 0..1, general capability
    coding: float = 0.6            # 0..1, code-specific
    speed: float = 0.5             # 0..1, higher = faster (tokens/s normalised)
    tags: list[str] = field(default_factory=list)

    def est_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.input_cost + completion_tokens * self.output_cost


def _load_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # corrupt file must not kill startup
        log.warning("failed to read %s: %s", path, exc)
        return {}


def _prices() -> dict:
    prices = _load_json(PRICES_FILE)
    if prices:
        return prices
    # ponytail: fall back to the copy litellm ships rather than shipping our own.
    try:
        import litellm

        return dict(litellm.model_cost)
    except Exception:
        return {}


def _provider_of(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    if model_id.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("gemini"):
        return "gemini"
    return "unknown"


class Catalog:
    def __init__(self, model_ids: list[str]):
        self.models: dict[str, ModelInfo] = {}
        self.reload(model_ids)

    def reload(self, model_ids: list[str]) -> None:
        prices = _prices()
        quality = _load_json(QUALITY_FILE)
        models: dict[str, ModelInfo] = {}
        for mid in model_ids:
            p = prices.get(mid) or prices.get(mid.split("/", 1)[-1]) or {}
            q = quality.get(mid) or quality.get(mid.split("/", 1)[-1]) or {}
            if not p and "input_cost_per_1m" not in q:
                log.warning("no price data for %s, and no override in quality.json", mid)
            # Providers the price sheet doesn't cover (NVIDIA NIM chat models,
            # anything self-hosted) can declare $/1M in quality.json instead.
            in_cost = float(p.get("input_cost_per_token") or 0.0)
            out_cost = float(p.get("output_cost_per_token") or 0.0)
            if "input_cost_per_1m" in q:
                in_cost = float(q["input_cost_per_1m"]) / 1_000_000
            if "output_cost_per_1m" in q:
                out_cost = float(q["output_cost_per_1m"]) / 1_000_000

            models[mid] = ModelInfo(
                id=mid,
                provider=p.get("litellm_provider") or _provider_of(mid),
                input_cost=in_cost,
                output_cost=out_cost,
                context=int(q.get("context") or p.get("max_input_tokens") or p.get("max_tokens") or 8192),
                max_output=int(p.get("max_output_tokens") or 4096),
                supports_tools=bool(q.get("supports_tools", p.get("supports_function_calling"))),
                quality=float(q.get("quality", 0.6)),
                coding=float(q.get("coding", q.get("quality", 0.6))),
                speed=float(q.get("speed", 0.5)),
                tags=list(q.get("tags", [])),
            )
        self.models = models

    def available(self) -> list[ModelInfo]:
        """Models we actually hold credentials for."""
        import litellm

        from .config import settings

        bases = settings.api_bases
        out = []
        for m in self.models.values():
            # A self-hosted OpenAI-compatible endpoint often needs no real key.
            if m.id in bases:
                out.append(m)
                continue
            try:
                missing = litellm.validate_environment(model=m.id).get("missing_keys")
            except Exception:
                missing = None
            if not missing:
                out.append(m)
        # ponytail: empty pool means no keys configured at all -- let the call
        # fail loudly downstream instead of silently returning nothing here.
        return out or list(self.models.values())

    def get(self, model_id: str) -> ModelInfo | None:
        return self.models.get(model_id)


_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        from .config import settings

        _catalog = Catalog(settings.model_pool)
    return _catalog
