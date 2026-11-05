# 3. Keep voc2vec over HuBERT

**Status:** accepted

## Context

On spoken words from TORGO, HuBERT beats voc2vec by about 9 points across any session (p = 0.016), and by about 12 on another day, with only 4 people. On non-verbal sounds from ReCANVo, voc2vec beats HuBERT by about 3 points (p = 0.039).

## Decision

Keep voc2vec as the default.

## Why

Each model wins on what it was trained on: HuBERT on speech, voc2vec on non-verbal sounds. Tink is for people making sounds that aren't words, so the non-verbal result is closer to the use case, even though the gain is smaller. Picking HuBERT because it has the bigger number would mean optimizing for my proxy instead of the real task.

The honest weakness: no benchmark has deliberate non-verbal sounds, so I can't measure this directly.

## Consequences

`TINK_MODEL=facebook/hubert-base-ls960` switches the server for someone who still has dysarthric speech. Cached embeddings are tagged by model, so switching can't mix them. A next experiment is fusing both models, which doubles the size and has to earn that.
