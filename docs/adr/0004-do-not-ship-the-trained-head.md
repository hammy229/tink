# 4. Don't ship the trained head

**Status:** accepted

## Context

Frozen embeddings with a centroid is a very simple model, so I tried to beat it. `evals/head.py` trains a linear projection with prototypical network episodes in PyTorch. Each episode is one person's own few-shot task, so what it learns is how to compare sounds, not any one person's meanings. The tested person is never in training, and neither is the person used for early stopping.

## Decision

Don't ship it.

## Why

Before running it I set a gate: it has to beat the shipped method on spontaneous sounds and on deliberate sounds, with the paired interval above zero. It gained 2.9 points on dysarthric speech and lost 1.0 on ReCANVo, so it failed.

I could have kept tuning until it passed, or reported only the dataset where it won. With 8 people, tuning against the same benchmark would just fit that benchmark.

## Consequences

The shipped system stays simple and trains nothing across people, which also means no one's voice has to be in a training set. The code stays in `evals/` so the result can be reproduced. A head could come back with a lot more people's data, which is what it probably needs.
