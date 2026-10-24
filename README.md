# Tink

Tinker Bell can't speak in words. She talks in chimes, and Peter understands her.

Tink does the same for people who can't be understood in words: people who have lost speech to ALS, stroke, or other conditions, and people who don't speak the language of the room. It learns the sounds one person makes (a hum, two clicks, a sigh) and says out loud what each one means, in the listener's language. Sounds belong to no language, so the same model serves both.

Built at HackTX 26 on [voc2vec](https://ieeexplore.ieee.org/document/10890672) (Koudounas et al., ICASSP 2025), a speech model pretrained on 125 hours of non-verbal vocalizations.

## Run it

Needs [uv](https://docs.astral.sh/uv/). The first run downloads the model (~360 MB).

```bash
uv sync
uv run uvicorn tink.app:app --reload
```

Open http://localhost:8000 in Chrome. Allow the microphone.

1. **Teach a sound:** type what the sound is and what it means (any language), then press **Record take** and make the sound. Do it about 5 times.
2. Teach at least 2 sounds. Tink compares sounds against each other, so it won't match with only one.
3. Press **Listen** (or space), make a sound, and Tink says what it means.
4. **Room noise:** press **Record 3 seconds of room** with nobody making a sound (do it 2 or 3 times, in the room you will demo in). Sounds that match the room are ignored.
5. **Urgent:** tick **Urgent** when teaching a sound like a cry of pain. Tink skips questions, fires at lower confidence, turns the screen red, and repeats the phrase until someone presses **I'm here**.
6. **Answer back:** type a reply (or tap a quick reply) and pick **Their language**. Tink translates it with Grok and says it to the person. Without a key it says it as typed.
7. Pick a language in **Speak in** (top right). With a Grok key, Tink translates the taught phrase into it and speaks it with a matching voice.

| Key | Does |
|---|---|
| Space | Listen |
| P | Present mode: the stage fills the screen for a projector |
| Esc | Leave present mode |

To let Grok word what Tink says, set a key first (optional; without it Tink says the taught phrase):

```bash
export XAI_API_KEY=...
export TINK_GROK_MODEL=grok-4.3   # list yours: curl https://api.x.ai/v1/models -H "Authorization: Bearer $XAI_API_KEY"
```

## How it works

```
mic clip ──► voc2vec scores ──┐
                              ├─► Bayesian posterior ──► JSON packet ──► Grok ──► spoken sentence
webcam ──► face cues ─────────┘   (time-of-day prior)
```

1. **Audio.** The browser records a clip, stops on its own when the sound ends, and sends a WAV. The backend resamples to 16 kHz mono and trims the silence.
2. **voc2vec.** Turns the clip into one vector (mean of one hidden layer over time). The released checkpoints have no classifier head, so there is no training. Each sound's vector is the average of its takes, and the clip is scored against each by cosine similarity after subtracting the mean of all takes.
3. **Augmentation and self-calibration** (`tink/audio.py`, `tink/store.py`). Each take also yields 4 variants (about a semitone up and down, two noise levels), so 5 recordings cover more of how a voice drifts. After every change Tink tests itself: each original take is held out with its own variants and matched against the rest. That fits the softmax temperature (floored at 0.04 so a few easy takes can't make it certain) and the "didn't catch it" threshold to this person, and the Sounds panel shows the result.
4. **Bayes** (`tink/bayes.py`). `P(sound | clip, cues, time) ∝ P(clip | sound) · P(cues | sound) · P(sound | time)`.
   - Audio likelihood: softmax of the scores. An "unknown" option sits at `TINK_THRESHOLD`, so weak matches become "I didn't catch that."
   - Time prior: Dirichlet-smoothed counts of confirmed uses per part of day. It learns every time someone presses **Right**.
   - Face cues: a sound taught with "smile" gets a boost when the camera sees a smile.
5. **Webcam** (`web/cues.js`). MediaPipe Face Landmarker runs in the browser and reports nod, shake, smile, frown, or eyes closed. Only those words leave the browser, never video.
6. **JSON packet → Grok** (`tink/llm.py`). The top candidates with probabilities, the time, recent phrases, and cue names. Grok words one short sentence, or asks an either/or question when the top two are close. If Grok fails or there is no key, Tink says the taught phrase. The UI shows the exact packet and offers "Say the taught phrase" whenever Grok rewords it.

## Risk test

Before tuning anything, check the core idea works on your own voices:

```bash
# teach 3-4 sounds x 5 takes in the app first (saved to data/takes/)
uv run python risk_test.py
```

It prints leave-one-out accuracy for every layer and a suggested threshold. Go if accuracy is 85% or more.

To check the paper's claim on your own recordings (voc2vec vs. general speech models, same clips, same test):

```bash
uv run python risk_test.py --compare
uv run python risk_test.py --augment    # accuracy with and without augmented variants
```

## Tests

```bash
uv run pytest -q
```

They use synthetic tones, so they need no recordings and never call Grok.

## Settings

| Env var | Default | What it does |
|---|---|---|
| `TINK_MODEL` | `alkiskoudounas/voc2vec-hubert-ls-pt` | Any voc2vec checkpoint |
| `TINK_LAYER` | `-1` | Hidden layer to pool (0 to 12) |
| `TINK_THRESHOLD` | `0.5` | "Didn't catch it" cutoff until calibration takes over |
| `TINK_DATA` | `data` | Where takes, `sounds.json`, and `usage.json` live |
| `TINK_TEMP` | `0.1` | Softmax temperature until calibration takes over (lower = more confident) |
| `TINK_AUGMENT` | `1` | Add 4 augmented variants per take (set `0` to turn off) |
| `XAI_API_KEY` | unset | Turns on Grok wording |
| `TINK_GROK_MODEL` | `grok-4.3` | xAI model id |

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/sounds` | | `{sounds: [{name, phrase, cues, takes}], threshold, cues, llm}` |
| POST | `/api/enroll` | form: `name`, `phrase`, `cues` (JSON list), `audio` (WAV) | `{name, takes}` |
| POST | `/api/recognize` | form: `audio` (WAV), `cues` (JSON list) | `{matched, name, p, say, ask, source, ranked, scores, packet}` |
| POST | `/api/confirm` | form: `name` | `{ok, usage}` |
| POST | `/api/reply` | form: `text`, `to` (language) | `{say, typed, to, source, packet}` |
| DELETE | `/api/sounds/{name}` | | `{ok}` |

## Layout

```
tink/audio.py    load, resample, trim
tink/embed.py    voc2vec embeddings
tink/store.py    enrolled sounds + scoring
tink/bayes.py    posterior over sounds (audio, time of day, face cues)
tink/llm.py      Grok wording with fallback
tink/app.py      FastAPI routes, serves web/
web/             frontend (plain HTML/CSS/JS, no build step); cues.js = MediaPipe face cues
risk_test.py     layer sweep + threshold; --compare runs the baseline models
tests/           pytest suite (synthetic audio)
PRODUCT.md       who Tink is for and what it must never claim
```
