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

## How it works

1. The browser records a clip and stops on its own when the sound ends, then sends a WAV.
2. The backend resamples to 16 kHz mono and trims the silence.
3. voc2vec turns the clip into one vector (mean of one hidden layer over time). The released checkpoints have no classifier head, so there is no training.
4. Each sound's vector is the average of its takes. A new clip matches the closest one by cosine similarity, after subtracting the mean of all takes. Below `TINK_THRESHOLD` it says "I didn't catch that."

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
| `TINK_DATA` | `data` | Where takes and `sounds.json` live |

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/sounds` | | `{sounds: [{name, phrase, takes}], threshold}` |
| POST | `/api/enroll` | form: `name`, `phrase`, `audio` (WAV) | `{name, takes}` |
| POST | `/api/recognize` | form: `audio` (WAV) | `{matched, name, phrase, score, scores}` |
| DELETE | `/api/sounds/{name}` | | `{ok}` |

## Layout

```
tink/audio.py    load, resample, trim
tink/embed.py    voc2vec embeddings
tink/store.py    enrolled sounds + matching
tink/app.py      FastAPI routes, serves web/
web/             frontend (plain HTML/CSS/JS, no build step)
risk_test.py     layer sweep + threshold
```
