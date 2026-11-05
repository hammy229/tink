"""Sliding-window rate limit for public mode, and how to tell who a client is."""
import time
from collections import defaultdict, deque
from collections.abc import Callable

MAX_TRACKED = 10_000  # clients remembered at once, so a scan from many addresses can't grow memory forever


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self.clock()
        if key not in self.hits and len(self.hits) >= MAX_TRACKED:
            self._prune(now)
        q = self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True

    def _prune(self, now: float) -> None:
        for key in [k for k, q in self.hits.items() if not q or now - q[-1] > self.window]:
            del self.hits[key]
        if len(self.hits) >= MAX_TRACKED:  # all active: forgetting some is better than running out of memory
            self.hits.clear()


def client_key(forwarded_for: str | None, peer: str | None, trust_proxy: bool) -> str:
    """Who to count a request against.

    Anyone can send X-Forwarded-For, so it only counts behind a proxy we run. A proxy appends the address it
    saw, so the last entry is the real one; everything before it came from the client and can be forged.
    """
    if trust_proxy and forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return peer or "?"
