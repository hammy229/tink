# How well does Tink work?

Short version: when someone teaches Tink 5 words in one sitting with 2 takes each and uses it on another day, it gets the right one 62% of the time, where guessing gets 20%. That number comes from only 4 speakers with dysarthria, so treat it as a first signal. On spontaneous sounds, where the label is a state of mind, it does much worse, 27% against 19%. Neither is reliable enough to depend on alone, which is why Tink asks instead of guessing and why a caregiver stays in the loop.

Exact numbers for everything below are in [results.md](results.md), which is generated from the raw results.

## The two datasets

**ReCANVo** (Narain and Johnson, 2021, CC BY 3.0) has 7,077 sounds recorded at home from 8 minimally speaking autistic people, where a caregiver labeled what each sound meant in the moment (frustrated, delighted, request, and so on). This is a hard test, as the label is the person's state, so a delighted sound and a self-talk sound can be the same sound.

**TORGO** (Rudzicz, Namasivayam, and Wolff, 2012) has speakers with dysarthria from cerebral palsy or ALS, plus matched controls, reading prompts. I use repeats of a single word as a stand-in for a deliberate signal, where the person means to make the same sound each time. Words aren't hums, so this is a proxy, and it's the closest public data I could find. TORGO is free for academic use, and its audio isn't in this repo.

Nobody has published deliberate signal sounds from nonspeaking people at scale. So neither dataset tests the exact use case, and only a study with real users can do that.

## Method

For each person: take a few examples of each sound, subtract the mean, average each sound into a centroid, and label new sounds by cosine to the nearest centroid. This is what the app does, except that the app also adds 4 augmented variants per take. [results.md](results.md) measures that separately.

- **TORGO:** 5 words at a time, 2 examples per word. Reported tested on any remaining take, and also taught in one sitting and tested only on other sessions, which is the number that matters.
- **ReCANVo:** 5 examples per meaning, tested on other sessions. Reported two ways, as a spontaneous recording session rarely has every meaning in it. One lets each meaning be taught on a different day. The other teaches everything in one sitting, which matches the app but covers only 2 or 3 meanings at a time, so chance is higher.

Accuracy is balanced (the average of per-class recall), so common sounds can't hide rare ones.

**The person is the unit of analysis.** Thousands of clips from 8 people are still 8 people, so every interval comes from resampling people. Comparisons are paired, where both sides see the same episodes. A method only counts as better if the paired interval is above zero and an exact sign-flip test gives p < 0.05, since a percentile bootstrap is too optimistic with so few people. One catch is that with 4 people the sign-flip test can't go below 0.125, so nothing can be declared better on the headline benchmark alone.

**Checks on the evaluation itself:**

- Shuffling the labels drops every benchmark to around chance, and the run fails if it doesn't. This only shows the scoring doesn't leak. Session and mic effects survive a shuffle, which is what the other-session benchmarks are for.
- The rewrite of the evaluation reproduced the original ReCANVo number exactly, 26.7%.
- Tests check that enrollment and test clips never overlap, and that "other session" really means another session.

## What I found

**How the benchmark is built changed the answer more than any model did.** My first cross-session version let each word be taught on a different day, and it scored 42%. A review pointed out that a session could then stand in for a word, and that the app teaches in one sitting. Fixing that gave 62%. So I had been underselling it, because of a mistake in the test.

**Sessions still matter.** On dysarthric speech, 77% across any session falls to 62% when tested on another day. These are different groups of people (7 and 4), so the gap is only a rough guide.

**A smaller model looked fine until the second dataset.** On ReCANVo, the convolutional front end alone (layer 0) scored 0.9 points better than all 12 layers, which would mean shipping a far smaller model. On dysarthric speech that same choice loses 4.8 points, and 8.4 on another day. If I had only tested on one dataset, I would have shipped the wrong thing. See [ADR 2](adr/0002-ship-the-full-depth-model.md).

**voc2vec isn't simply the best model.** It beats HuBERT and wav2vec2 on non-verbal sounds, which is what it was trained on, but HuBERT beats it by 9 to 12 points on spoken words. See [ADR 3](adr/0003-keep-voc2vec-over-hubert.md).

**The trained head didn't ship.** I trained a small projection with prototypical network episodes in PyTorch, leaving out the tested person each time. It gained 2.9 points on dysarthric speech and lost 1.0 on ReCANVo. I set the rule before running it: it has to win on both. See [ADR 4](adr/0004-do-not-ship-the-trained-head.md).

**Learning from "Right" got shipped and then removed.** The idea was that when a caregiver confirms a sound, that clip joins the sound. My first simulation showed a gain, so I built it. A review found the flaw: the confirmed clips and the scored clips came from one random pool, so a clip could help a near copy of itself. In time order, with some wrong confirmations, it barely helps ReCANVo (+0.6) and hurts TORGO by 7.0 points, p = 0.004. See [ADR 5](adr/0005-do-not-learn-from-confirmed-sounds.md).

**LDA is the best lead for next time.** Shrinkage LDA, fixed at 0.5 and not tuned, gains 3.6 to 9.7 points on TORGO and loses about 1 on ReCANVo. It fails the same rule, so it doesn't ship, but it needs no training across people, so it's the first thing I'd test on new data.

**Augmentation isn't proven.** It gains 4.3 points on the headline benchmark (p = 0.125, with 4 people), and on the others it does nothing or loses about 1. I left it on, as the headline case is the closest to real use and it only costs a little time when teaching, but this is a judgment call and the data doesn't settle it.

**Tink's confidence is usable but not trustworthy on its own.** Fitting the temperature to each person cuts calibration error on most benchmarks, though not on controls across sessions. On ReCANVo it still says about 48% when it's right 27% of the time. I also tried conformal prediction, where Tink would offer a set of options with a coverage guarantee. Against a 90% target, coverage was 67% on dysarthric speech across sessions and 53% on controls, as the guarantee assumes new sounds look like the enrolled ones, and sessions break that. So Tink doesn't advertise a guarantee it can't keep.

**Asking is worth a lot.** Tink asks about one meaning at a time, best guess first, so the number of questions is the rank of the right answer. On the headline benchmark the right answer is within 2 questions 84% of the time, up from 62% on the first try.

## What this can't tell you

- It hasn't been tested with a nonspeaking person using it.
- TORGO's words are a proxy for deliberate sounds. Real hums and clicks could be easier, as they can be chosen to be distinct, or harder, as there's less in them to recognize.
- 4 to 8 people per benchmark is a small sample, and people vary a lot. Layer, method, and model were all compared on these same people, so anything found here is exploratory until it's tested on new data.
- It doesn't measure the "unknown" option: every test sound is one of the taught sounds, so how well Tink ignores a sound it was never taught isn't tested. The room noise role isn't benchmarked either.
- The app's self-test holds out takes from the same sitting, so it runs higher than real use on another day. The app says so.

The next real test is recording deliberate sounds from people who use them to communicate.

## Reproduce

```bash
# ReCANVo: https://doi.org/10.5281/zenodo.5786860   TORGO: https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
uv run --group eval python -m evals.embed recanvo path/to/ReCANVo
uv run --group eval python -m evals.embed torgo path/to/TORGO
uv run --group eval python -m evals.run --recanvo path/to/ReCANVo --torgo path/to/TORGO
uv run --group eval python -m evals.render
```

Everything is seeded, so the same data gives the same numbers.
