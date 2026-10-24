"""Tink API. Run: uv run uvicorn tink.app:app --reload  then open http://localhost:8000

Each turn: mic clip -> voc2vec scores -> Bayesian posterior (time of day, webcam cues)
-> JSON packet -> Grok -> what Tink says.
"""
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import audio, bayes, llm
from .store import THRESHOLD, Store

app = FastAPI(title="Tink")
store = Store()
recent: deque[dict] = deque(maxlen=5)  # what Tink said lately, for the LLM's context


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


def _fallback(ranked: list[dict]) -> dict:
    """What Tink says with no LLM: the phrase, an either/or question, or ask to try again."""
    phrase = {n: s["phrase"] for n, s in store.sounds.items()}
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    if top["name"] == bayes.UNKNOWN:
        return {"say": "I didn't catch that. Can you make the sound again?", "ask": True}
    if top["p"] < 0.6 and second and second["name"] != bayes.UNKNOWN:
        return {"say": f'Do you mean "{phrase[top["name"]]}" or "{phrase[second["name"]]}"?', "ask": True}
    return {"say": phrase[top["name"]], "ask": False}


@app.get("/api/sounds")
def sounds():
    return {
        "sounds": store.all(),
        "threshold": THRESHOLD,
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
    takes = store.add_take(name, phrase, await _read(audio_file), _cues(cues))
    return {"name": name, "takes": takes}


@app.post("/api/recognize")
async def recognize(cues: str = Form("[]"), audio_file: UploadFile = File(..., alias="audio")):
    scores = store.scores(await _read(audio_file))
    if not scores:
        raise HTTPException(409, "Teach Tink at least one sound first.")
    now = datetime.now()
    post = bayes.posterior(scores, THRESHOLD, store.usage, store.cues(), _cues(cues), now.hour)
    ranked = post["ranked"]

    phrase = {n: s["phrase"] for n, s in store.sounds.items()}
    packet = {
        "time": now.strftime("%H:%M"),
        "part_of_day": post["part_of_day"],
        "candidates": [
            {"sound": r["name"], "means": phrase.get(r["name"], "(no match)"), "p": r["p"]} for r in ranked[:3]
        ],
        "webcam_cues": post["cues"],
        "recent": list(recent),
    }
    out = await llm.compose(packet)
    source = "grok" if out else "phrase"
    out = out or _fallback(ranked)
    if not out["ask"]:
        recent.append({"time": packet["time"], "said": out["say"]})

    top = ranked[0]
    return {
        "matched": top["name"] != bayes.UNKNOWN,
        "name": top["name"] if top["name"] != bayes.UNKNOWN else None,
        "p": top["p"],
        "say": out["say"],
        "ask": out["ask"],
        "source": source,
        "ranked": ranked,
        "scores": scores,
        "packet": packet,
    }


@app.post("/api/confirm")
def confirm(name: str = Form(...)):
    """Someone pressed "Right": teach the time-of-day prior that this sound gets used now."""
    if not store.confirm(name, datetime.now().hour):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True, "usage": store.usage.get(name, {})}


@app.delete("/api/sounds/{name}")
def forget(name: str):
    if not store.delete(name):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")
