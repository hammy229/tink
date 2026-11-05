"""Measure what one Tink server does under concurrent use, end to end over HTTP.

    uv run uvicorn tink.app:app --port 8799  (with TINK_DATA pointing at an empty folder)
    uv run python -m scripts.loadtest --base http://127.0.0.1:8799

Teaches a few synthetic sounds, then runs closed-loop clients at rising concurrency: each client sends its next clip
when the last one returns. Also checks that /healthz stays fast while the model is busy, which is the point of
running inference off the event loop.
"""
import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np

from tests.helpers import hiss, tone

OUT = Path(__file__).parent.parent / "evals" / "results" / "load.json"


def _pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 1)


async def _client(c: httpx.AsyncClient, n: int, seed: int, latencies: list[float], errors: list[int]) -> None:
    for i in range(n):
        clip = tone(140 + (seed + i) % 40, seed=seed * 1000 + i)
        t0 = time.perf_counter()
        r = await c.post("/api/recognize", files={"audio": ("a.wav", clip)}, timeout=120)
        latencies.append((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            errors.append(r.status_code)


async def _probe(c: httpx.AsyncClient, stop: asyncio.Event, out: list[float]) -> None:
    while not stop.is_set():
        t0 = time.perf_counter()
        await c.get("/healthz", timeout=30)
        out.append((time.perf_counter() - t0) * 1000)
        await asyncio.sleep(0.02)


async def run(base: str, levels: list[int], per_client: int) -> dict:
    results = []
    async with httpx.AsyncClient(base_url=base) as c:
        for i in range(5):
            for name, clip in (("Hum", tone(140, seed=i)), ("Hiss", hiss(seed=i)), ("High", tone(900, seed=i))):
                r = await c.post("/api/enroll", data={"name": name, "phrase": name},
                                 files={"audio": ("a.wav", clip)}, timeout=120)
                r.raise_for_status()
        for level in levels:
            latencies: list[float] = []
            errors: list[int] = []
            health: list[float] = []
            stop = asyncio.Event()
            probe = asyncio.create_task(_probe(c, stop, health))
            t0 = time.perf_counter()
            await asyncio.gather(*[_client(c, per_client, s, latencies, errors) for s in range(level)])
            wall = time.perf_counter() - t0
            stop.set()
            await probe
            row = {"clients": level, "requests": len(latencies), "errors": len(errors),
                   "throughput_rps": round(len(latencies) / wall, 2), "p50_ms": _pct(latencies, 50),
                   "p95_ms": _pct(latencies, 95), "p99_ms": _pct(latencies, 99),
                   "healthz_p95_ms": _pct(health, 95), "healthz_max_ms": round(max(health), 1)}
            results.append(row)
            print(row, flush=True)
    return {"machine": f"{platform.system()} {platform.machine()}", "clip_seconds": 0.8, "sounds": 3,
            "takes_per_sound": 5, "note": "one profile, so turns queue behind its lock by design", "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8799")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--per-client", type=int, default=30)
    args = ap.parse_args()
    report = asyncio.run(run(args.base, args.levels, args.per_client))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
