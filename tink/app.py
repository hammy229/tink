"""Tink API. Run: uv run uvicorn tink.app:app --reload  then open http://localhost:8000

Each turn: mic clip -> voc2vec scores -> Bayesian posterior (time of day, webcam cues)
-> JSON packet -> Grok -> what Tink says, in the listener's language.

Every route is `async def` on purpose: none of them await while touching the Store, so on
uvicorn's single event loop they run one at a time and can't interleave mid-mutation.
"""
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import audio, bayes, llm
from .store import MAX_TEXT, MIN_SOUNDS, THRESHOLD, Store

app = FastAPI(title="Tink")
store = Store()  # loads the model and every take at startup
recent: deque[dict] = deque(maxlen=5)  # what Tink said lately, for the LLM's context
ASK_BELOW = 0.6  # top probability under this means Tink asks instead of saying


async def _read(upload: UploadFile):
    try:
        return audio.load(await upload.read())
    except Exception as e:
        raise HTTPException(400, f"Could not read audio: {e}") from e


def _cues(raw: str) -> list[str]:
    try:
        out = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [c for c in out if c in bayes.CUES] if isinstance(out, list) else []


def _fallback(ranked: list[dict], phrase: dict[str, str]) -> dict:
    """What Tink says with no LLM: the phrase, an either/or question, or ask to try again."""
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    if top["name"] == bayes.UNKNOWN or top["name"] not in phrase:
        return {"say": "I didn't catch that. Can you make the sound again?", "ask": True}
    if top["p"] < ASK_BELOW and second and second["name"] in phrase:
        return {"say": f'Do you mean "{phrase[top["name"]]}" or "{phrase[second["name"]]}"?', "ask": True}
    return {"say": phrase[top["name"]], "ask": False}


@app.get("/api/sounds")
async def sounds():
    return {
        "sounds": store.all(),
        "threshold": THRESHOLD,
        "min_sounds": MIN_SOUNDS,
        "cues": bayes.CUES,
        "llm": {"on": bool(llm.KEY), "model": llm.MODEL},
    }


@app.post("/api/enroll")
async def enroll(
    name: str = Form(...),
    phrase: str = Form(...),
    cues: str = Form("[]"),
    audio_file: UploadFile = File(..., alias="audio"),
):
    name, phrase = name.strip(), phrase.strip()
    if not name or not phrase:
        raise HTTPException(400, "Both the sound name and the phrase are required.")
    if len(name) > MAX_TEXT or len(phrase) > MAX_TEXT:
        raise HTTPException(400, f"Keep the sound name and phrase under {MAX_TEXT} characters.")
    wav = await _read(audio_file)
    if audio.is_silent(wav):
        raise HTTPException(400, "Tink didn't hear a sound in that take. Move closer to the mic and try again.")
    takes = store.add_take(name, phrase, wav, _cues(cues))
    return {"name": name, "takes": takes}


@app.post("/api/recognize")
async def recognize(
    cues: str = Form("[]"),
    speak_in: str = Form("English"),
    audio_file: UploadFile = File(..., alias="audio"),
):
    wav = await _read(audio_file)
    phrase = {n: s["phrase"] for n, s in store.sounds.items()}  # snapshot: a Forget mid-request can't break this turn
    scores = store.scores(wav)
    if not scores:
        raise HTTPException(409, f"Teach Tink at least {MIN_SOUNDS} sounds first, so it can tell them apart.")
    now = datetime.now()
    post = bayes.posterior(scores, THRESHOLD, store.usage, store.cues(), _cues(cues), now.hour)
    ranked = post["ranked"]

    packet = {
        "time": now.strftime("%H:%M"),
        "part_of_day": post["part_of_day"],
        "speak_in": speak_in.strip()[:30] or "English",
        "candidates": [
            {"sound": r["name"], "means": phrase.get(r["name"], "(no match)"), "p": r["p"]} for r in ranked[:3]
        ],
        "webcam_cues": post["cues"],
        "recent": list(recent),
    }
    out = await llm.compose(packet)
    source = "grok" if out else "phrase"
    out = out or _fallback(ranked, phrase)

    top = ranked[0]
    matched = top["name"] != bayes.UNKNOWN and top["name"] in phrase
    if not matched or top["p"] < ASK_BELOW:
        out["ask"] = True  # never let wording turn an unsure guess into a statement
    if not out["ask"]:
        recent.append({"time": packet["time"], "said": out["say"]})
    return {
        "matched": matched,
        "name": top["name"] if matched else None,
        "p": top["p"],
        "say": out["say"],
        "ask": out["ask"],
        "source": source,
        "taught": phrase.get(top["name"]),
        "ranked": ranked,
        "scores": scores,
        "packet": packet,
    }


@app.post("/api/confirm")
async def confirm(name: str = Form(...)):
    """Someone pressed "Right": teach the time-of-day prior that this sound gets used now."""
    if not store.confirm(name, datetime.now().hour):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True, "usage": store.usage.get(name, {})}


@app.delete("/api/sounds/{name:path}")
async def forget(name: str):
    if not store.delete(name):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")
