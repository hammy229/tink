# Model card: Tink's sound recognizer

## What it is

Few-shot recognition of one person's sounds. A frozen speech encoder turns each clip into one vector, and a small per-person model on top compares new sounds with the ones that person taught it. Nothing is trained across people in the shipped system.

- **Encoder:** [voc2vec-hubert-ls-pt](https://huggingface.co/alkiskoudounas/voc2vec-hubert-ls-pt) (Koudounas et al., ICASSP 2025, Apache 2.0), last layer, mean pooled, L2 normalized.
- **Per person:** centered cosine to the average of each sound's takes and their augmented variants, softmax with a temperature fitted by leave-one-take-out, an "unknown" option, and a time-of-day prior.
- **Browser build:** the same encoder exported to ONNX with int8 dynamic quantization, 122 MB, pinned by SHA-256.

## Intended use

Helping someone who can make sounds but not words be understood by people around them, with a caregiver present. It suggests what a sound means. It asks when it's unsure.

## Not for

- Being the only way someone can call for help. Tink has an urgent alarm, and the alarm can miss.
- Any medical decision, diagnosis, or tracking of disease.
- Recognizing anyone other than the person who taught it. It learns from that one person's examples.

## Data

No training data in the shipped system beyond the examples the person gives it. Evaluated on ReCANVo and TORGO, described in [accuracy.md](accuracy.md).

## Performance

See [results.md](results.md). In short, out of 5 options where chance is 20%: about 62% on dysarthric speech taught in one sitting and tested on another day (4 people), and 77% across any session (7 people). On spontaneous sounds it gets 27% against 19%.

## Groups

Accuracy varies a lot between people, and [results.md](results.md) lists each one. Benchmarks hold 4 to 8 people each, which is too few to say anything about differences by sex, age, diagnosis, or severity, so I don't report them. TORGO doesn't state which speakers have ALS and which have cerebral palsy, so I can't report ALS on its own.

## Known limits

- Not tested with a nonspeaking person using it. Words stand in for deliberate sounds in the main benchmark.
- Accuracy falls between sessions.
- It's overconfident on spontaneous sounds. Don't read its percentage as a promise.
- The "unknown" option isn't benchmarked, so how often it wrongly fires on a sound it was never taught isn't known.
- A background noise that matches a taught sound can fire it. Teaching the room's noise is meant to reduce this.

## Privacy

In the browser build, audio never leaves the device, and an end-to-end test checks that no request carries any. On a server, clips and banked voices are stored under the person's profile, and logs contain numbers only, never phrases or sound names. Forgetting a sound deletes its clips and embeddings.

## How a new model gets shipped

1. `scripts/export_onnx.py` refuses a build whose vectors drift from PyTorch (mean cosine below 0.99, worst clip below 0.98).
2. `evals/onnx_gate.py` re-scores the benchmarks with the exact file and fails on a loss above 1 point.
3. The file's hash goes in `web/engine/config.js`, the browser checks it before loading, and CI fails unless that hash has a passing report.

To roll back, point `config.js` at the previous URL and hash. Vectors are tagged with the model that made them, so a change never mixes them.
