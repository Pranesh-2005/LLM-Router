"""LiteLLM execution layer: retries with exponential backoff + provider fallback."""

from __future__ import annotations

import logging

from .config import settings
from .schemas import RouteRequest, Usage

log = logging.getLogger(__name__)


def _messages(req: RouteRequest) -> list[dict]:
    msgs = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.append({"role": "user", "content": req.query})
    return msgs


MIN_REASONING_BUDGET = 512


def _text_of(resp) -> tuple[str, str | None]:
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        # Reasoning models emit hidden reasoning tokens that come out of
        # max_tokens. With a small budget the whole allowance is spent thinking
        # and `content` arrives empty even though usage is non-zero.
        text = (getattr(msg, "reasoning_content", None) or "").strip()
    return text, getattr(resp.choices[0], "finish_reason", None)


async def complete(model: str, req: RouteRequest) -> tuple[str, Usage, str]:
    """Returns (text, usage, model_actually_used)."""
    import litellm

    fallbacks = [m for m in settings.fallbacks if m != model]
    kwargs = dict(
        model=model,
        messages=_messages(req),
        temperature=req.temperature,
        timeout=settings.request_timeout,
        # litellm retries transient errors (429/5xx/timeouts) with exponential
        # backoff and honours Retry-After, then moves down the fallback chain.
        num_retries=settings.num_retries,
        fallbacks=fallbacks or None,
        api_base=settings.api_bases.get(model),
    )
    resp = await litellm.acompletion(max_tokens=req.max_tokens, **kwargs)
    text, finish = _text_of(resp)

    if not text and finish == "length" and req.max_tokens < MIN_REASONING_BUDGET:
        # One retry with enough headroom to finish thinking AND answer. Returning
        # an empty string to the caller is never an acceptable outcome.
        log.info("empty content from %s at max_tokens=%d, retrying with %d",
                 model, req.max_tokens, MIN_REASONING_BUDGET)
        resp = await litellm.acompletion(max_tokens=MIN_REASONING_BUDGET, **kwargs)
        text, finish = _text_of(resp)

    used = getattr(resp, "model", model) or model
    if not text:
        raise RuntimeError(
            f"{used} returned empty content (finish_reason={finish}); raise max_tokens"
        )

    u = getattr(resp, "usage", None)
    cost = 0.0
    try:
        cost = float(resp._hidden_params.get("response_cost") or 0.0)
    except Exception:
        pass
    usage = Usage(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        total_tokens=getattr(u, "total_tokens", 0) or 0,
        cost_usd=round(cost, 6),
    )
    return text, usage, used


async def stream(model: str, req: RouteRequest):
    import litellm

    fallbacks = [m for m in settings.fallbacks if m != model]
    resp = await litellm.acompletion(
        model=model,
        messages=_messages(req),
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        timeout=settings.request_timeout,
        num_retries=settings.num_retries,
        fallbacks=fallbacks or None,
        api_base=settings.api_bases.get(model),
        stream=True,
    )
    async for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
