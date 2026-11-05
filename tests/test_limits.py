"""Rate limiting for public mode."""
from tink.limits import RateLimiter, client_key


def test_rate_limit_blocks_then_forgives():
    t = [0.0]
    limit = RateLimiter(2, window=60, clock=lambda: t[0])
    assert limit.allow("a") and limit.allow("a") and not limit.allow("a")
    assert limit.allow("b")
    t[0] = 61
    assert limit.allow("a")


def test_rate_limit_memory_is_bounded(monkeypatch):
    monkeypatch.setattr("tink.limits.MAX_TRACKED", 10)
    t = [0.0]
    limit = RateLimiter(5, window=60, clock=lambda: t[0])
    for i in range(1000):
        t[0] = i * 100.0
        limit.allow(str(i))
    assert len(limit.hits) <= 11


def test_forged_forwarded_header_cannot_dodge_the_limit():
    assert client_key("1.2.3.4", "9.9.9.9", trust_proxy=False) == "9.9.9.9"
    assert client_key("forged, 5.6.7.8", "10.0.0.1", trust_proxy=True) == "5.6.7.8"
    assert client_key(None, None, trust_proxy=True) == "?"
