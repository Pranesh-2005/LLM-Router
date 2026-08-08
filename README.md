# LLM Router

One endpoint. It reads the query, decides what kind of task it is, and sends it
to the model that handles that task best at the lowest sensible cost.

```
POST /v1/route   {"query": "..."}   ->   {"output": "...", "decision": {...}}
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add at least one provider key
python scripts/update_catalog.py     # refresh prices/context windows (optional, has a bundled copy)
uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/v1/route \
  -H 'content-type: application/json' \
  -H 'Idempotency-Key: 7c1f-demo' \
  -d '{"query":"Fix this traceback: TypeError: NoneType is not subscriptable"}' | jq
```

Interactive docs at `http://localhost:8000/docs`.

Offline checks (no keys, no network): `python tests/test_router.py`

Live end-to-end test against real providers: `python app.py` — preflights every
configured model, then exercises routing, guardrail, a real completion, the
semantic cache, idempotency replay, cross-provider fallback and the rate
limiter. `python app.py --preflight` just checks reachability;
`python app.py --serve` boots uvicorn instead.

Docker: `docker compose up --build` (the compose file also runs a weekly catalog refresh).

---

## The two agents

### Agent 1 — Sentinel (`app/agents.py`)

Guardrail **and** intent classification in one pass, because they read the same
text and splitting them would double the latency for no gain.

1. **Regex hard-block** — prompt injection, system-prompt/credential exfil,
   weapons synthesis, malware requests. Zero tokens, zero latency. Deliberately
   narrow: it only catches the unambiguous cases.
2. **Heuristic intent** — weighted regex signals per intent (`coding`,
   `writing`, `summarisation`, `translation`, `reasoning`, `general`) plus a
   complexity estimate. Confidence is high only when the winner is both strong
   *and* clear of the runner-up.
3. **LLM classifier** — runs **only** when the heuristic is under-confident
   (`< 0.8`). Cheap model, `response_format=json_object`, 6 s timeout, 160 max
   tokens. Result goes into a semantic cache keyed on the query.
4. **Fail-open on intent, fail-closed on safety.** If the classifier call
   fails, the heuristic intent is used — worst case is a mediocre model pick.
   The safety pass never depends on the network, because step 1 already ran.

Most production traffic never reaches step 3, which is where the latency budget
is protected.

### Agent 2 — Router (`app/agents.py`)

Deterministic scoring, **no LLM call**, sub-millisecond. Every candidate model
gets one score:

```
score = wq * quality_axis + wc * cheapness + ws * speed
```

- **`quality_axis`** is `coding` for coding intents, general `quality` otherwise.
- **`cheapness`** is the min-max-normalised estimated cost for *this* request
  (prompt chars / 4 + requested `max_tokens`, priced from the live catalog).
- **`speed`** is normalised output throughput.

Weights come from two tables:

| intent | quality | cost | speed | axis |
|---|---|---|---|---|
| coding | 0.60 | 0.15 | 0.25 | coding |
| reasoning | 0.65 | 0.15 | 0.20 | quality |
| writing | 0.50 | 0.20 | 0.30 | quality |
| summarisation | 0.25 | 0.40 | 0.35 | quality |
| translation | 0.30 | 0.35 | 0.35 | quality |
| general | 0.40 | 0.30 | 0.30 | quality |

then multiplied by the caller's `priority` bias (`cheap` / `balanced` / `quality`)
and renormalised. Rationale: summarisation and translation are near-solved, so a
small fast model is the right answer and cost dominates. Coding and multi-step
reasoning are where model quality still changes the outcome, so quality dominates.

**Hard filters run before scoring** (they are constraints, not preferences):
- context window must hold prompt + completion,
- `supports_function_calling` when the classifier flagged tool use,
- a **quality floor** by complexity (`high` → ≥ 0.75) so a "cheap" priority can
  never drop a hard problem onto a nano model.

`POST /v1/route/preview` returns the decision and the top-5 candidates with
scores without calling any provider — useful for tuning the weight tables.

---

## System design

**Idempotency** — `Idempotency-Key` header. First request stores its response
under the key (24 h). A replay returns the stored body with `cached:
"idempotent"`. A **concurrent** duplicate waits on the in-flight request rather
than firing a second paid call. Reusing a key with a different body is `409` —
silently returning the wrong cached answer is worse than an error.

**Semantic cache** — exact SHA-256 hit first (O(1)), then a nearest-neighbour
scan over cached embeddings; `cosine ≥ 0.93` is a hit. Namespaced per
`model:max_tokens`, so a cache entry can never leak across models. Only for
`temperature ≤ 0.1` — a caller asking for temperature is asking for variety, so
replaying an answer would be wrong.

The default embedder is a hashed bag-of-words + char-4-gram vector: pure Python,
~50 µs, no network hop. It catches reordering, casing and typos, not true
synonymy. Swap `embed()` in `app/cache.py` for a real embedding model if you
need that — the cache only needs `embed(text) -> dict[int, float]`.

**Retries & backoff** — LiteLLM's `num_retries=3` retries 429/5xx/timeouts with
exponential backoff and honours `Retry-After`; when the model is still failing,
`fallbacks=[...]` moves to a different provider entirely, and the response
records which model actually served it. Not reimplemented here — the SDK's
version already handles per-provider error taxonomies.

**Rate limiting** — token bucket per API key (client IP when unauthenticated),
`60 rpm` / burst `20`. Returns `429` with a real `Retry-After`.

**Latency** — the fast path is: regex classify → arithmetic route → upstream
call. No classifier call, no embedding service, no database. Router overhead is
well under a millisecond; everything above the provider call is in-process.

**Extensibility** — adding a provider is one line in `ROUTER_MODELS` plus its
API key. LiteLLM resolves the provider from the id prefix, prices come from the
catalog, and models with no credentials are filtered out at routing time. No
code change, no new class.

### OpenAI-compatible providers

Anything with a known prefix works as-is — id in `ROUTER_MODELS`, key in `.env`:

```
deepseek/deepseek-chat        together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo
groq/llama-3.3-70b-versatile  openrouter/anthropic/claude-sonnet-5
ollama/llama3                 hosted_vllm/my-model
```

Self-hosted `/v1` servers (vLLM, LM Studio, LiteLLM proxy) need a base URL:

```bash
ROUTER_MODELS=gpt-5.4-mini,claude-sonnet-5,openai/qwen2.5-coder-32b
MODEL_API_BASES=openai/qwen2.5-coder-32b=http://localhost:8000/v1
```

A model listed in `MODEL_API_BASES` skips the credential check (local servers
often have no key) and keeps its real `$0` cost instead of being treated as
unpriced. Give it a `data/quality_overrides.json` entry, otherwise it routes on
the default score of 0.6.

---

## Keeping the catalog current

`scripts/update_catalog.py`

- pulls LiteLLM's public price sheet (ids, input/output price, context window,
  tool support) — the same file LiteLLM itself ships,
- merges OpenRouter's `/api/v1/models` for anything not in it (no key needed),
- writes `data/model_prices.json` and `data/quality.json`,
- reports **price drift** on the routed models so a provider price change is
  visible instead of silently reshaping every routing decision,
- `--check` reports drift and writes nothing (use it in CI).

**Benchmarks.** LMArena, Artificial Analysis and SWE-bench have no free, stable,
machine-readable endpoint. Rather than scrape something that breaks monthly,
scores live in `data/quality_overrides.json` as hand-maintained numbers, and the
updater **never** overwrites them. For a model it has never seen it derives a
starting score from output price tier and name hints (`mini`/`flash`/`haiku` →
fast, `opus`/`pro` → strong), tags it `auto-derived`, and prints it for review.
Price is a mediocre proxy for capability, but it is an honest one, and it is
clearly labelled as provisional.

Run it weekly (`compose` does this) or on deploy.

---

## API

| endpoint | purpose |
|---|---|
| `POST /v1/route` | route + execute. `Idempotency-Key` optional. `stream:true` → SSE |
| `POST /v1/route/preview` | routing decision only, no provider call |
| `GET /v1/models` | catalog with live prices and scores |
| `GET /v1/stats` | request / cache-hit / block counters |
| `GET /healthz` | liveness |

Request body: `query`, `system?`, `priority?` (`cheap`/`balanced`/`quality`),
`max_tokens`, `temperature`, `model_hint?` (bypass routing), `stream`, `cache`.

Response carries the full `decision` (chosen model, score, estimated cost, top-5
candidates, a plain-text reason) and `classification`, so every route is
auditable after the fact.

Auth: set `API_KEYS` to a comma list and send `X-API-Key` / `Authorization:
Bearer`. Empty means open (dev default).

---

## Known limits

- **Single node.** Cache, rate-limit buckets and the idempotency store are
  in-process (`app/cache.py`). Two workers means two of everything. Swap those
  three classes for Redis before scaling out — the call sites are four lines.
- **Lexical, not semantic, cache** by default. See above.
- **Quality scores are hand-maintained.** They are the one input to routing that
  no API gives you for free.
