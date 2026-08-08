from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import llm
from .agents import Router, Sentinel
from .cache import IdempotencyStore, SemanticCache
from .catalog import get_catalog
from .config import settings
from .ratelimit import TokenBucket
from .schemas import Classification, Decision, RouteRequest, RouteResponse, Usage

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("router")

sentinel = Sentinel()
router_agent: Router | None = None
semantic_cache = SemanticCache(
    maxsize=settings.semantic_cache_size,
    ttl=settings.semantic_cache_ttl,
    threshold=settings.semantic_cache_threshold,
)
idempotency = IdempotencyStore(ttl=settings.idempotency_ttl)
bucket = TokenBucket(settings.rate_limit_rpm, settings.rate_limit_burst)

STATS = {"requests": 0, "semantic_hits": 0, "idempotent_hits": 0, "blocked": 0, "errors": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import litellm

    litellm.drop_params = True          # provider-unsupported params get dropped, not 400'd
    litellm.suppress_debug_info = True
    global router_agent
    router_agent = Router()
    log.info("catalog loaded: %s", ", ".join(get_catalog().models))
    yield


app = FastAPI(
    title="LLM Router",
    version="1.0.0",
    description="Intent-aware LLM router with guardrails, semantic cache and idempotency.",
    lifespan=lifespan,
)


def _client_id(request: Request, api_key: str | None) -> str:
    if api_key:
        return f"key:{api_key[-8:]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path in ("/healthz", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    api_key = request.headers.get("x-api-key") or (
        request.headers.get("authorization", "").removeprefix("Bearer ").strip() or None
    )
    allowed = settings.allowed_keys
    if allowed and api_key not in allowed:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    ok, retry_after = bucket.take(_client_id(request, api_key))
    if not ok:
        return JSONResponse(
            {"error": "rate_limited", "detail": f"retry in {retry_after:.1f}s"},
            status_code=429,
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )

    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Latency-Ms"] = str(int((time.perf_counter() - start) * 1000))
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    cat = get_catalog()
    return {
        "models": [
            {
                "id": m.id,
                "provider": m.provider,
                "input_cost_per_1m": round(m.input_cost * 1_000_000, 4),
                "output_cost_per_1m": round(m.output_cost * 1_000_000, 4),
                "context": m.context,
                "quality": m.quality,
                "coding": m.coding,
                "speed": m.speed,
                "supports_tools": m.supports_tools,
            }
            for m in cat.models.values()
        ]
    }


@app.get("/v1/stats")
async def stats():
    return {
        **STATS,
        "semantic_cache_entries": len(semantic_cache._exact),
        "models": len(get_catalog().models),
    }


@app.post("/v1/route/preview", response_model=Decision)
async def preview(req: RouteRequest):
    """Routing decision only -- no upstream call. Useful for tuning weights."""
    cls = await sentinel.classify(req)
    if not cls.allowed:
        raise HTTPException(422, detail=cls.reason or "blocked")
    assert router_agent
    return router_agent.route(req, cls)


@app.post("/v1/route", response_model=RouteResponse)
async def route(
    req: RouteRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    STATS["requests"] += 1
    t0 = time.perf_counter()

    if len(req.query) > settings.max_query_chars:
        raise HTTPException(413, detail=f"query exceeds {settings.max_query_chars} chars")

    fp = IdempotencyStore.fingerprint(req.model_dump_json())
    owner = True
    if idempotency_key:
        try:
            cached, owner = await idempotency.begin(idempotency_key, fp)
        except IdempotencyStore.Conflict:
            raise HTTPException(409, detail="Idempotency-Key reused with a different body")
        if cached is not None:
            STATS["idempotent_hits"] += 1
            out = RouteResponse(**cached)
            out.cached = "idempotent"
            return out
        if not owner:
            raise HTTPException(409, detail="concurrent request with same key still in flight")

    try:
        resp = await _handle(req, t0)
    except HTTPException:
        if idempotency_key and owner:
            idempotency.finish(idempotency_key, None)
        raise
    except Exception as exc:
        STATS["errors"] += 1
        if idempotency_key and owner:
            idempotency.finish(idempotency_key, None)
        log.exception("route failed")
        raise HTTPException(502, detail=f"upstream error: {exc}") from exc

    if idempotency_key and owner and not isinstance(resp, StreamingResponse):
        idempotency.finish(idempotency_key, resp.model_dump())
    return resp


async def _handle(req: RouteRequest, t0: float):
    # 1. Agent 1 -- guardrail + intent
    cls = await sentinel.classify(req)
    if not cls.allowed:
        STATS["blocked"] += 1
        raise HTTPException(422, detail=cls.reason or "request blocked by guardrail")

    # 2. Agent 2 -- model selection
    assert router_agent
    if req.model_hint:
        info = get_catalog().get(req.model_hint)
        decision = Decision(
            model=req.model_hint,
            provider=info.provider if info else "unknown",
            intent=cls.intent, priority=req.priority or settings.default_priority,  # type: ignore[arg-type]
            score=1.0, est_cost_usd=0.0, candidates=[], reason="model_hint override",
        )
    else:
        decision = router_agent.route(req, cls)

    # 3. Semantic cache (deterministic requests only -- a temperature>0 request
    #    is asking for variety, so replaying an answer would be wrong).
    cache_ns = f"{decision.model}:{req.max_tokens}"
    cacheable = (
        settings.semantic_cache_enabled and req.cache and req.temperature <= 0.1 and not req.stream
    )
    if cacheable:
        hit = semantic_cache.get(cache_ns, (req.system or "") + "\n" + req.query)
        if hit is not None:
            payload, sim = hit
            STATS["semantic_hits"] += 1
            return RouteResponse(
                id=f"rt_{uuid.uuid4().hex[:16]}",
                output=payload["output"],
                decision=decision,
                classification=cls,
                usage=Usage(**payload["usage"]),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                cached="semantic",
            )

    # 4. Execute (retries + fallbacks inside)
    if req.stream:
        return StreamingResponse(_sse(decision, cls, req), media_type="text/event-stream")

    text, usage, used = await llm.complete(decision.model, req)
    if used and used != decision.model:
        decision.reason += f" | served by fallback {used}"

    if cacheable:
        semantic_cache.set(
            cache_ns,
            (req.system or "") + "\n" + req.query,
            {"output": text, "usage": usage.model_dump()},
        )

    return RouteResponse(
        id=f"rt_{uuid.uuid4().hex[:16]}",
        output=text,
        decision=decision,
        classification=cls,
        usage=usage,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        cached=False,
    )


async def _sse(decision: Decision, cls: Classification, req: RouteRequest):
    yield "event: decision\ndata: " + json.dumps(decision.model_dump()) + "\n\n"
    try:
        async for delta in llm.stream(decision.model, req):
            yield "data: " + json.dumps({"delta": delta}) + "\n\n"
    except Exception as exc:  # stream already started -- report inline
        yield "event: error\ndata: " + json.dumps({"error": str(exc)}) + "\n\n"
    yield "event: done\ndata: {}\n\n"
