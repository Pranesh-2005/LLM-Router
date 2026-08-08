#!/usr/bin/env python
"""Refresh the model catalog: names, context windows, prices, quality scores.

Sources
  1. LiteLLM price sheet (authoritative for id / price / context / capabilities)
     https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
  2. OpenRouter /api/v1/models (fills gaps, no key needed)
  3. data/quality_overrides.json -- hand-maintained benchmark scores.

Why hand-maintained scores: LMArena / Artificial Analysis / SWE-bench have no
free, stable, machine-readable endpoint. Rather than scrape something that
breaks weekly, the script keeps human numbers and only *derives* a starting
score for models it has never seen (price tier + name are decent proxies).

    python scripts/update_catalog.py            # refresh everything
    python scripts/update_catalog.py --check    # show drift, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "model_prices.json"
QUALITY = DATA / "quality.json"
OVERRIDES = DATA / "quality_overrides.json"

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

FAST_HINTS = ("mini", "flash", "haiku", "lite", "small", "instant", "8b", "turbo")
STRONG_HINTS = ("opus", "gpt-5", "o1", "o3", "sonnet-4", "ultra", "pro", "405b", "70b")


def fetch_json(url: str, timeout: float = 30.0) -> dict | None:
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  ! {url} failed: {exc}", file=sys.stderr)
        return None


def merge_openrouter(prices: dict, orx: dict) -> int:
    added = 0
    for m in orx.get("data", []):
        mid = f"openrouter/{m['id']}"
        if mid in prices:
            continue
        p = m.get("pricing") or {}
        try:
            inp, out = float(p.get("prompt", 0)), float(p.get("completion", 0))
        except (TypeError, ValueError):
            continue
        prices[mid] = {
            "litellm_provider": "openrouter",
            "input_cost_per_token": inp,
            "output_cost_per_token": out,
            "max_input_tokens": m.get("context_length") or 8192,
            "max_output_tokens": (m.get("top_provider") or {}).get("max_completion_tokens") or 4096,
            "supports_function_calling": "tools" in (m.get("supported_parameters") or []),
        }
        added += 1
    return added


def derive_scores(mid: str, meta: dict) -> dict:
    """Starting scores for a model nobody has benchmarked by hand yet.

    Output price is the least-bad public proxy for capability tier, and the
    name usually announces the speed tier. Both get overwritten the moment a
    real number lands in quality_overrides.json.
    """
    out_cost_per_1m = float(meta.get("output_cost_per_token") or 0.0) * 1_000_000
    if out_cost_per_1m >= 30:
        q = 0.92
    elif out_cost_per_1m >= 10:
        q = 0.85
    elif out_cost_per_1m >= 3:
        q = 0.75
    elif out_cost_per_1m >= 0.5:
        q = 0.66
    else:
        q = 0.55

    low = mid.lower()
    speed = 0.5
    if any(h in low for h in FAST_HINTS):
        speed, q = 0.85, min(q, 0.78)
    if any(h in low for h in STRONG_HINTS):
        speed, q = min(speed, 0.45), max(q, 0.88)

    return {
        "quality": round(q, 2),
        "coding": round(min(0.98, q + 0.02), 2),
        "speed": speed,
        "tags": ["auto-derived"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    ap.add_argument("--models", default="", help="comma list to score; default = ROUTER_MODELS/.env")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)

    print("fetching LiteLLM price sheet...")
    prices = fetch_json(LITELLM_URL)
    if not prices:
        print("could not fetch prices; keeping existing file", file=sys.stderr)
        prices = json.loads(PRICES.read_text(encoding="utf-8")) if PRICES.exists() else {}
    prices.pop("sample_spec", None)

    print("fetching OpenRouter catalog...")
    orx = fetch_json(OPENROUTER_URL)
    if orx:
        print(f"  + {merge_openrouter(prices, orx)} openrouter models")

    # which models do we actually score?
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        sys.path.insert(0, str(ROOT))
        from app.config import settings  # noqa: E402

        wanted = settings.model_pool

    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8")) if OVERRIDES.exists() else {}
    previous = json.loads(QUALITY.read_text(encoding="utf-8")) if QUALITY.exists() else {}

    quality, new, missing = {}, [], []
    for mid in wanted:
        meta = prices.get(mid) or prices.get(mid.split("/", 1)[-1])
        if meta is None:
            missing.append(mid)
            meta = {}
        if mid in overrides:
            quality[mid] = overrides[mid]
        elif mid in previous and "auto-derived" not in previous[mid].get("tags", []):
            quality[mid] = previous[mid]
        else:
            quality[mid] = derive_scores(mid, meta)
            if mid not in previous:
                new.append(mid)

    # price drift report
    drift = []
    for mid in wanted:
        meta = prices.get(mid) or prices.get(mid.split("/", 1)[-1]) or {}
        old = {}
        if PRICES.exists():
            oldall = json.loads(PRICES.read_text(encoding="utf-8"))
            old = oldall.get(mid) or oldall.get(mid.split("/", 1)[-1]) or {}
        for field in ("input_cost_per_token", "output_cost_per_token"):
            if old.get(field) not in (None, meta.get(field)):
                drift.append(f"  ~ {mid} {field}: {old[field]} -> {meta.get(field)}")

    if missing:
        print(f"  ! no price data for: {', '.join(missing)}")
    if new:
        print(f"  + new models scored (review data/quality.json): {', '.join(new)}")
    for line in drift:
        print(line)

    if args.check:
        print("--check: nothing written")
        return 1 if (drift or missing) else 0

    PRICES.write_text(json.dumps(prices, indent=0, sort_keys=True), encoding="utf-8")
    QUALITY.write_text(json.dumps(quality, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {PRICES.relative_to(ROOT)} ({len(prices)} models)")
    print(f"wrote {QUALITY.relative_to(ROOT)} ({len(quality)} routed models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
