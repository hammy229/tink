# 5. Don't learn from confirmed sounds

**Status:** accepted, after the opposite decision was built and reversed

## Context

Sounds change between sessions, and every time a caregiver presses **Right**, they hand Tink a correctly labeled new example. Adding that clip to the sound seemed like free accuracy with no training.

## What happened

My first simulation played test clips in random order, added correct guesses, and scored the second half. It gained 2.6 points on ReCANVo and 2.3 on dysarthric speech, so I built it, in the server and the browser, with tests.

A review then found three problems with that simulation:

1. The confirmed clips and the scored clips came from one random pool, so a clip could help a near copy of itself.
2. Clips weren't in time order, so it said nothing about using Tink later on.
3. The simulated caregiver never made a mistake.

I fixed all three in `evals/fewshot.py:run_adaptive`: clips play in recording order, the earlier half gets confirmed, only the later half is scored, and a wrong guess gets confirmed 10% of the time.

## Decision

Remove the feature. **Right** only teaches the time-of-day prior, as it did before.

## Why

In the corrected simulation it gains 0.6 points on ReCANVo (p = 0.10) and loses 7.0 points on TORGO, [-9.9, -4.4], p = 0.004. Those are the people closest to the real use case, so it would have made Tink worse for them. I didn't look for a variant that passes, as with these few people I'd just be fitting the benchmark.

I haven't worked out why it hurts. My guess is that centering and centroids go off balance when easy sounds pick up extra clips and hard ones don't, but I haven't tested that.

## Consequences

- The experiment stays in `evals/` so it can be reproduced, and a class-balanced version could be tried on new data.
- The lesson I'm keeping: my first simulation gave me the answer I wanted, and I shipped on it. That's why these numbers now get reviewed before anything is built on them.
