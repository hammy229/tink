"""Model calls take real time. They must not freeze the server, and one person's turns must not interleave."""
import asyncio
import importlib
import threading
import time

import httpx
import numpy as np
import pytest

from tests.helpers import tone

SLOW = 0.4


@pytest.fixture()
def slow_app(tmp_path, monkeypatch):
    monkeypatch.setenv("TINK_DATA", str(tmp_path))
    from tink import app as app_module
    from tink import profiles as profiles_module
    from tink import store as store_module

    importlib.reload(store_module)
    importlib.reload(profiles_module)
    importlib.reload(app_module)
    monkeypatch.setattr(app_module.llm, "KEY", None)

    active = {"now": 0, "peak": 0, "slow": False}
    guard = threading.Lock()

    def embed(wav):
        with guard:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        if active["slow"]:
            time.sleep(SLOW)
        with guard:
            active["now"] -= 1
        rng = np.random.default_rng(int(abs(wav).sum() * 1000) % 2**32)
        v = rng.normal(size=8)
        return (v / np.linalg.norm(v)).astype(np.float32)

    monkeypatch.setattr(store_module, "embed", embed)
    return app_module, active


def _client(app_module):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://test")


async def _teach(c):
    for i in range(2):
        for name, freq in (("Hum", 140), ("Hiss", 900)):
            r = await c.post("/api/enroll", data={"name": name, "phrase": "x"},
                             files={"audio": ("a.wav", tone(freq, seed=i))})
            assert r.status_code == 200


def test_server_answers_other_requests_during_inference(slow_app):
    app_module, active = slow_app

    async def go():
        async with _client(app_module) as c:
            await _teach(c)
            active["slow"] = True
            turn = asyncio.create_task(c.post("/api/recognize", files={"audio": ("a.wav", tone(142))}))
            await asyncio.sleep(0.05)  # let the turn reach the model
            t0 = time.perf_counter()
            health = await c.get("/healthz")
            waited = time.perf_counter() - t0
            assert not turn.done(), "the turn should still be inside the model"
            assert (await turn).status_code == 200
            return health.status_code, waited

    status, waited = asyncio.run(go())
    assert status == 200
    assert waited < SLOW / 2


def test_one_persons_turns_never_overlap(slow_app):
    app_module, active = slow_app

    async def go():
        async with _client(app_module) as c:
            await _teach(c)
            active["peak"] = 0
            active["slow"] = True
            rs = await asyncio.gather(*[c.post("/api/recognize", files={"audio": ("a.wav", tone(142, seed=i))})
                                        for i in range(3)])
            assert {r.status_code for r in rs} == {200}

    asyncio.run(go())
    assert active["peak"] == 1
