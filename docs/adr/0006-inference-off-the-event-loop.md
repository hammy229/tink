# 6. Keep inference off the event loop

**Status:** accepted

## Context

The routes were `async def` and called torch directly. That made the code easy to reason about, since requests couldn't interleave, but one person's model call froze every other request on the server, including health checks.

## Decision

- Model calls and disk writes run on worker threads through `run_in_threadpool`.
- Each profile has an `asyncio.Lock`, so one person's turns stay in order while other people's keep going.
- One global lock sits around the model, as MPS isn't thread-safe and parallel inference on the same cores gains little.
- Profile loading has its own lock, as two `Store` objects for one person would lose writes.

## Why not the other options

- **Multiple processes:** each one loads a 360 MB model, and profile state lives in memory.
- **A task queue with workers:** a lot of machinery for a single-box app. It's the right next step if this ever needed more than one machine.

## Consequences

`tests/test_concurrency.py` checks both promises: `/healthz` answers during inference, and one person's turns never overlap. `scripts/loadtest.py` measures it end to end. Throughput is capped by the single model lock, and latency grows with the queue, which is fine for one household and is the first thing to change for real scale.
