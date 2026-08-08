"""Token-bucket rate limiter, keyed on API key (falls back to client IP)."""

from __future__ import annotations

import time
from collections import OrderedDict


class TokenBucket:
    def __init__(self, rpm: int, burst: int, maxsize: int = 10000):
        self.rate = rpm / 60.0
        self.burst = max(burst, 1)
        self.maxsize = maxsize
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def take(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens >= cost:
            self._buckets[key] = (tokens - cost, now)
            self._buckets.move_to_end(key)
            self._evict()
            return True, 0.0
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        self._evict()
        return False, (cost - tokens) / self.rate

    def _evict(self) -> None:
        while len(self._buckets) > self.maxsize:
            self._buckets.popitem(last=False)
