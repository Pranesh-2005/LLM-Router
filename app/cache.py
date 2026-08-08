"""TTL/LRU cache, semantic cache and idempotency store.

ponytail: all in-process (OrderedDict + time). Single-node only. Swap the three
classes for Redis when you run more than one worker -- the call sites are 4 lines.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, maxsize: int = 1024, ttl: float = 900.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._d.get(key)
        if item is None:
            return None
        exp, val = item
        if exp < time.monotonic():
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return val

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._d[key] = (time.monotonic() + (ttl or self.ttl), value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)


# --- embeddings -------------------------------------------------------------
# ponytail: hashed bag-of-words + char 4-grams, pure python, ~50us per query and
# zero network hops. It catches paraphrase-by-reordering and typos, not true
# synonymy. Set EMBEDDER=litellm-style real embeddings if you need that -- the
# cache only needs `embed(text) -> dict[int, float]`.

_WORD = re.compile(r"[a-z0-9']+")
_DIMS = 4096


def embed(text: str) -> dict[int, float]:
    t = text.lower().strip()
    vec: dict[int, float] = {}
    words = _WORD.findall(t)
    for w in words:
        h = hash(w) % _DIMS
        vec[h] = vec.get(h, 0.0) + 1.0
    squashed = " ".join(words)
    for i in range(len(squashed) - 3):
        h = hash(squashed[i : i + 4]) % _DIMS
        vec[h] = vec.get(h, 0.0) + 0.5
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class SemanticCache:
    """Exact-hash first (O(1)); nearest-neighbour scan only on miss."""

    def __init__(self, maxsize: int = 2000, ttl: float = 900.0, threshold: float = 0.93):
        self.threshold = threshold
        self.ttl = ttl
        self.maxsize = maxsize
        self._exact = TTLCache(maxsize, ttl)
        self._entries: OrderedDict[str, tuple[float, dict[int, float], Any]] = OrderedDict()

    def _sweep(self) -> None:
        now = time.monotonic()
        dead = [k for k, (exp, _, _) in self._entries.items() if exp < now]
        for k in dead:
            self._entries.pop(k, None)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def get(self, namespace: str, text: str) -> tuple[Any, float] | None:
        key = f"{namespace}:{hashlib.sha256(text.strip().lower().encode()).hexdigest()}"
        hit = self._exact.get(key)
        if hit is not None:
            return hit, 1.0
        self._sweep()
        v = embed(text)
        best, best_sim = None, 0.0
        for k, (_, ev, val) in self._entries.items():
            if not k.startswith(namespace + ":"):
                continue
            sim = cosine(v, ev)
            if sim > best_sim:
                best, best_sim = val, sim
        if best is not None and best_sim >= self.threshold:
            return best, best_sim
        return None

    def set(self, namespace: str, text: str, value: Any) -> None:
        key = f"{namespace}:{hashlib.sha256(text.strip().lower().encode()).hexdigest()}"
        self._exact.set(key, value)
        self._entries[key] = (time.monotonic() + self.ttl, embed(text), value)
        self._entries.move_to_end(key)
        self._sweep()


class IdempotencyStore:
    """Replay the stored response for a repeated Idempotency-Key.

    In-flight duplicates wait on the first request instead of firing a second
    (and being billed twice). Body mismatch on a reused key raises Conflict.
    """

    class Conflict(Exception):
        pass

    def __init__(self, ttl: float = 86400.0, maxsize: int = 10000):
        self._done = TTLCache(maxsize, ttl)
        self._inflight: dict[str, asyncio.Event] = {}
        self._fingerprints = TTLCache(maxsize, ttl)

    @staticmethod
    def fingerprint(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()

    async def begin(self, key: str, fp: str) -> tuple[Any | None, bool]:
        """Returns (cached_response, is_owner). Owner must call finish()."""
        known_fp = self._fingerprints.get(key)
        if known_fp is not None and known_fp != fp:
            raise self.Conflict(key)

        cached = self._done.get(key)
        if cached is not None:
            return cached, False

        ev = self._inflight.get(key)
        if ev is not None:
            try:
                await asyncio.wait_for(ev.wait(), timeout=120)
            except asyncio.TimeoutError:
                return None, True
            return self._done.get(key), False

        self._fingerprints.set(key, fp)
        self._inflight[key] = asyncio.Event()
        return None, True

    def finish(self, key: str, response: Any | None) -> None:
        if response is not None:
            self._done.set(key, response)
        ev = self._inflight.pop(key, None)
        if ev is not None:
            ev.set()
