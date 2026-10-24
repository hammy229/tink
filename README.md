# Tink

Tinker Bell can't speak in words. She talks in chimes, and Peter understands her.

Tink does the same for people who are losing their speech to ALS. It learns the sounds one person can still make (a hum, two clicks, a sigh) and says out loud what each one means. While the person can still speak, they bank their own voice: they record each phrase, and when Tink recognizes the sound it plays their recording instead of a computer voice. Everything runs on the laptop, and nothing is sent to any AI service by default.

Built at HackTX 26 on [voc2vec](https://ieeexplore.ieee.org/document/10890672) (Koudounas et al., ICASSP 2025), a speech model pretrained on 125 hours of non-verbal vocalizations.

## Run it

Needs [uv](https://docs.astral.sh/uv/). The first run downloads the model (~360 MB).

```bash
uv sync
uv run uvicorn tink.app:app --reload
```

Open http://localhost:8000 in Chrome. Allow the microphone.

1. **Teach a sound:** type what the sound is and what it means (the phrase can be in the person's own language), then press **Record take** and make the sound. Do it about 5 times.
2. Teach at least 2 sounds. Tink compares sounds against each other, so it won't match with only one.
3. **Bank their voice:** while the person can still speak, press **Record them saying it** (in the Sounds list or the Teach panel). The phrase appears on the paper screen to read aloud. From then on, Tink plays their recording when it recognizes that sound. **Say it again** and the urgent alarm use it too. If there is no recording (or Mute voice is on), Tink uses the computer voice.
4. Press **Listen** (or space), make a sound, and Tink says what it means.
5. **Room noise:** press **Record 3 seconds of room** with nobody making a sound (do it 2 or 3 times, in the room you will demo in). Sounds that match the room are ignored.
6. **Yes and no:** teach a sound marked **It means yes** and one marked **It means no**. When Tink is unsure, it asks about one meaning at a time and the person answers with those sounds.
7. **Urgent:** tick **Urgent** when teaching a sound like a cry of pain. Tink skips questions, fires at lower confidence, turns the screen red, and repeats the phrase until someone presses **I'm here**.
8. **Answer back:** type a reply (or tap a quick reply) and Tink speaks it aloud to the person, as typed. (Translation needs the optional LLM below.)

| Key | Does |
|---|---|
| Space | Listen |
| P | Present mode: the stage fills the screen for a projector |
| Esc | Leave present mode |

Optional and off by default: `tink/llm.py` can reword phrases and translate replies with an LLM if you set `XAI_API_KEY`. Without it, no text or audio goes anywhere.

## Hosting

Set `TINK_PUBLIC=1` to give each person a profile with a private link, so one deployment can serve several people. A Docker image is included (see `docs/deploy.md`).

## How it works

```
mic clip ──► voc2vec scores ──┐
                              ├─► Bayesian posterior ──► their banked voice, or the taught phrase ──► spoken
webcam ──► face cues ─────────┘   (time-of-day prior)
```

1. **Audio.** The browser records a clip, stops on its own when the sound ends, and sends a WAV. The backend resamples to 16 kHz mono and trims the silence.
2. **voc2vec.** Turns the clip into one vector (mean of one hidden layer over time). The released checkpoints have no classifier head, so there is no training. Each sound's vector is the average of its takes, and the clip is scored against each by cosine similarity after subtracting the mean of all takes.
3. **Augmentation and self-calibration** (`tink/audio.py`, `tink/store.py`). Each take also yields 4 variants (about a semitone up and down, two noise levels), so 5 recordings cover more of how a voice drifts. After every change Tink tests itself: each original take is held out with its own variants and matched against the rest. That fits the softmax temperature (floored at 0.04 so a few easy takes can't make it certain) and the "didn't catch it" threshold to this person, and the Sounds panel shows "right on N of M takes".
4. **Bayes** (`tink/bayes.py`). `P(sound | clip, cues, time) ∝ P(clip | sound) · P(cues | sound) · P(sound | time)`.
   - Audio likelihood: softmax of the scores. An "unknown" option sits at `TINK_THRESHOLD`, so weak matches become "I didn't catch that."
   - Time prior: Dirichlet-smoothed counts of confirmed uses per part of day. It learns every time someone presses **Right**.
   - Face cues: a sound taught with "smile" gets a boost when the camera sees a smile.
5. **Webcam** (`web/cues.js`). MediaPipe Face Landmarker runs in the browser and reports nod, shake, smile, frown, or eyes closed. Only those words leave the browser, never video.
6. **Speaking.** If the person banked that phrase, the browser plays their recording (`/api/voice/{name}`). Otherwise it speaks the taught phrase with the browser's voice. When the top two meanings are close, Tink asks instead, and the person answers with their own yes and no sounds.

## ## Risk test

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

They use synthetic tones, so they need no recordings and never call any LLM.

## Settings

| Env var | Default | What it does |
|---|---|---|
| `TINK_MODEL` | `alkiskoudounas/voc2vec-hubert-ls-pt` | Any voc2vec checkpoint |
| `TINK_LAYER` | `-1` | Hidden layer to pool (0 to 12) |
| `TINK_THRESHOLD` | `0.5` | "Didn't catch it" cutoff until calibration takes over |
| `TINK_DATA` | `data` | Where takes, `sounds.json`, and `usage.json` live |
| `TINK_TEMP` | `0.1` | Softmax temperature until calibration takes over (lower = more confident) |
| `TINK_AUGMENT` | `1` | Add 4 augmented variants per take (set `0` to turn off) |
| `XAI_API_KEY` | unset | Optional LLM, off by default. Turns on rewording and translation |
| `TINK_GROK_MODEL` | `grok-4.3` | Optional LLM, off by default: model id |
| `TINK_PUBLIC` | unset | Set to `1` to host: per-person profiles with private links |
| `TINK_MAX_PROFILES` | `500` | Cap on profiles when hosting |

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/sounds` | | `{sounds: [{name, phrase, cues, takes}], threshold, cues, llm}` |
| POST | `/api/enroll` | form: `name`, `phrase`, `cues` (JSON list), `audio` (WAV) | `{name, takes}` |
| POST | `/api/recognize` | form: `audio` (WAV), `cues` (JSON list) | `{matched, name, p, say, ask, source, ranked, scores, packet}` |
| POST | `/api/voice` | form: `name`, `audio` (WAV) | `{ok, name}` (banks the person saying this sound's phrase) |
| GET | `/api/voice/{name}` | | WAV of the banked voice, or 404 |
| POST | `/api/confirm` | form: `name` | `{ok, usage}` |
| POST | `/api/reply` | form: `text`, `to` (language) | `{say, typed, to, source, packet}` |
| DELETE | `/api/sounds/{name}` | | `{ok}` |

## Layout

```
tink/audio.py    load, resample, trim
tink/embed.py    voc2vec embeddings
tink/store.py    enrolled sounds, banked voices, scoring
tink/bayes.py    posterior over sounds (audio, time of day, face cues)
tink/llm.py      optional LLM wording (off by default)
tink/profiles.py per-person profiles for hosting
tink/app.py      FastAPI routes, serves web/
web/             frontend (plain HTML/CSS/JS, no build step); cues.js = MediaPipe face cues
risk_test.py     layer sweep + threshold; --compare runs the baseline models
tests/           pytest suite (synthetic audio)
PRODUCT.md       who Tink is for and what it must never claim
```
