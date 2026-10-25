# Accuracy on real nonspeaking people

Short version: on real sounds from 8 minimally speaking people, Tink guesses the meaning right **27%** of the time, against **19%** for guessing at random. That is above chance for 6 of the 8 people, but far too low to rely on. It is also close to the **30%** that a classifier trained on everything else that person recorded got, so the limit is mostly in the data, not in learning from only 5 examples.

## The data

[ReCANVo](https://doi.org/10.5281/zenodo.5786860) (Narain and Johnson, 2021, CC BY 3.0): 7,077 sounds recorded at home from 8 minimally speaking autistic people. A caregiver labeled each sound with what it meant in the moment: frustrated, delighted, request, self-talk, social, yes, and so on. Per person, I kept meanings with at least 20 clips, which leaves 4 to 6 meanings each.

This is a harder test than Tink's intended use. Tink assumes a person makes a *deliberate* sound for each meaning ("two clicks means water"). ReCANVo's sounds are spontaneous, and the label is the person's state, so a delighted sound and a self-talk sound can be the same sound. Nobody has recorded deliberate signal sounds from nonspeaking people at scale, so this is the closest real data there is.

## Method

Same as the app: for each person, take 5 examples per meaning, subtract the mean, average each meaning into a centroid, and label every new sound with the nearest centroid (cosine). Then score on sounds it didn't see.

- **Other sessions** (the main number): the 5 examples come from one recording session, and the test sounds only from other sessions. Sounds from one sitting are alike, so mixing sessions would inflate the score. This is also how Tink would be used: teach once, use on later days.
- **Random**: examples and test sounds drawn at random, ignoring sessions. Optimistic.

Each number is balanced accuracy (the average of per-meaning accuracy, so common meanings don't dominate), averaged over 50 random splits per person and then over people. Chance is 1 / number of meanings. Shuffling the labels drops every person to chance, which checks that the scoring doesn't leak.

## Results

| Model | Other sessions, app layer | Other sessions, best layer* | Random, app layer | Random, best layer* |
|---|---|---|---|---|
| voc2vec (what Tink uses) | **27%** | 27% | **33%** | 34% |
| HuBERT | 24% | 28% | 26% | 35% |
| wav2vec2 | 23% | 26% | 26% | 35% |
| Chance | 19% | 19% | 17% | 17% |

\*Layer picked on the other 7 people, then scored on the held-out person, so the layer choice never sees the person it's scored on.

Per person (voc2vec, other sessions): P01 25%, P02 26%, P03 35%, P05 26%, P06 24%, P08 23%, P11 18%, P16 37%. P02 and P11 are at chance.

Follow-up checks on voc2vec (one-off runs, not in the script):

| Check | Accuracy |
|---|---|
| Only answer the 25% of sounds Tink is surest of (other sessions) | 31% |
| Only answer the 10% it is surest of | 35% |
| 10 examples per meaning (random split) | 38% |
| 20 examples per meaning (random split) | 42% |
| Logistic regression trained on all of a person's other sessions | 30% |

## What this means

- **voc2vec's advantage is that it works without tuning.** At the layer the app uses, it beats the two general speech models by 3 to 7 points. Once each model gets its best layer, all three are about the same. The ICASSP paper's advantage for voc2vec doesn't clearly carry over to this data.
- **More examples help a little; confidence barely helps.** Going from 5 to 20 examples adds about 7 points. Answering only when sure adds a few points, so the "ask when unsure" behavior won't rescue accuracy on sounds like these.
- **Spontaneous sounds don't carry their meaning well.** A classifier trained on far more data does only slightly better than 5 examples. If Tink works, it will be for deliberate, distinct sounds, which this dataset can't test. The next real test is recording deliberate sounds from people who use them to communicate.

## Reproduce

```bash
# download ReCANVo.zip from https://doi.org/10.5281/zenodo.5786860 and unzip it
uv run python eval_recanvo.py embed path/to/ReCANVo   # about 30 min on an M-series Mac, caches every layer
uv run python eval_recanvo.py score path/to/ReCANVo
```
