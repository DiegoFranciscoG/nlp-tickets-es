"""Rate limiting en memoria con ventana deslizante (OWASP API4).

Suficiente para una sola instancia (Render free). Con varias réplicas habría que moverlo a Redis.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_identities: int = 10_000,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_identities = max_identities
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, identity: str) -> tuple[bool, int]:
        """Devuelve (permitido, segundos hasta el próximo cupo)."""
        now = self._clock()
        with self._lock:
            if identity not in self._hits and len(self._hits) >= self._max_identities:
                self._evict(now)
            hits = self._hits.setdefault(identity, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False, max(1, int(self._window - (now - hits[0])) + 1)
            hits.append(now)
            return True, 0

    def _evict(self, now: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or now - v[-1] >= self._window]
        for key in stale:
            del self._hits[key]
