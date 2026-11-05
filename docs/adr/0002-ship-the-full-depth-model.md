# 2. Ship the full depth model

**Status:** accepted

## Context

122 MB is a big download. Earlier layers should work almost as well, and cutting layers makes the file smaller and inference faster. On ReCANVo, layer 0 (no transformer at all) even scored 0.9 points better than layer 12.

## Decision

Keep all 12 layers and use one model everywhere.

## Why

The second dataset changed the answer. On dysarthric speech, layer 0 loses 4.8 points (p = 0.047), and 8.4 when tested on another day. Layer 6 loses 1.5, which isn't clearly real (p = 0.094), but it isn't a free saving either. The size saving is also smaller than it sounds: the 6 layer int8 build is 79 MB, only 35% smaller, since the front end doesn't shrink with the layers.

So half depth risks some accuracy for a modest saving, and it would mean the browser and server run different models, so results measured on one wouldn't apply to the other. Accuracy matters more than 43 MB on a one-time download.

## Consequences

The download stays large, and `evals/onnx_gate.py` re-scores the exact shipped file, so the published numbers apply to what people run. Exports with `--layers` still work, if a smaller build is ever worth it.
