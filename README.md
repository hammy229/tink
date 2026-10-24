# Tink

Tinker Bell can't speak in words. She talks in chimes, and Peter understands her.

Tink does the same for people who don't use words. It learns the sounds one person makes (a hum, two clicks, a sigh) and says what each one means out loud.

Built at HackTX 26 on [voc2vec](https://ieeexplore.ieee.org/document/10890672) (Koudounas et al., ICASSP 2025), a speech model pretrained on 125 hours of non-verbal vocalizations.

## Run it

Needs [uv](https://docs.astral.sh/uv/). The first run downloads the model (~360 MB).

```bash
uv sync
uv run uvicorn tink.app:app --reload
```

Open http://localhost:8000 in Chrome. Allow the microphone.

1. **Teach a sound:** type what the sound is and what Tink should say, then press **Record take** and make the sound. Do it 5 times.
2. Teach at least 2 sounds.
3. Press **Listen** (or space), make a sound, and Tink says the phrase.

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
3. **Bayes** (`tink/bayes.py`). `P(sound | clip, cues, time) ∝ P(clip | sound) · P(cues | sound) · P(sound | time)`.
   - Audio likelihood: softmax of the scores. An "unknown" option sits at `TINK_THRESHOLD`, so weak matches become "I didn't catch that."
   - Time prior: Dirichlet-smoothed counts of confirmed uses per part of day. It learns every time someone presses **Right**.
   - Face cues: a sound taught with "smile" gets a boost when the camera sees a smile.
4. **Webcam** (`web/cues.js`). MediaPipe Face Landmarker runs in the browser and reports nod, shake, smile, frown, or eyes closed. Only those words leave the browser, never video.
5. **JSON packet → Grok** (`tink/llm.py`). The top candidates with probabilities, the time, recent phrases, and cue names. Grok words one short sentence, or asks an either/or question when the top two are close. If Grok fails or there is no key, Tink says the taught phrase. The UI shows the exact packet and offers "Say the taught phrase" whenever Grok rewords it.

## Risk test

Before tuning anything, check the core idea works on your own voices:

```bash
# teach 3-4 sounds x 5 takes in the app first (saved to data/takes/)
uv run python risk_test.py
```

It prints leave-one-out accuracy for every layer and a suggested threshold. Go if accuracy is 85% or more.

## Settings

| Env var | Default | What it does |
|---|---|---|
| `TINK_MODEL` | `alkiskoudounas/voc2vec-hubert-ls-pt` | Any voc2vec checkpoint |
| `TINK_LAYER` | `-1` | Hidden layer to pool (0 to 12) |
| `TINK_THRESHOLD` | `0.5` | Minimum centered cosine to count as a match |
| `TINK_DATA` | `data` | Where takes, `sounds.json`, and `usage.json` live |
| `TINK_TEMP` | `0.1` | Softmax temperature on audio scores (lower = more confident) |
| `XAI_API_KEY` | unset | Turns on Grok wording |
| `TINK_GROK_MODEL` | `grok-4.3` | xAI model id |

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/sounds` | | `{sounds: [{name, phrase, cues, takes}], threshold, cues, llm}` |
| POST | `/api/enroll` | form: `name`, `phrase`, `cues` (JSON list), `audio` (WAV) | `{name, takes}` |
| POST | `/api/recognize` | form: `audio` (WAV), `cues` (JSON list) | `{matched, name, p, say, ask, source, ranked, scores, packet}` |
| POST | `/api/confirm` | form: `name` | `{ok, usage}` |
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
risk_test.py     layer sweep + threshold
```
