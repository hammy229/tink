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
from .store import MAX_TEXT, MIN_SOUNDS, ROLES, THRESHOLD, Store

app = FastAPI(title="Tink")
store = Store()  # loads the model and every take at startup
recent: deque[dict] = deque(maxlen=5)  # what Tink said lately, for the LLM's context
ASK_BELOW = 0.6  # top probability under this means Tink asks instead of saying
DIALOG_SECONDS = 45  # how long a yes/no question stays open
ANSWER_MIN = 0.6  # a yes or no sound must be at least this sure to answer a question
URGENT_MIN = 0.4  # urgent sounds alert at lower confidence: a missed alarm costs more than a false one
# When Tink is unsure and the person has a yes sound and a no sound, Tink asks about one meaning
# at a time: {"options": [sound names, best first], "i": index being asked about, "at": datetime}.
pending: dict | None = None


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
    role: str = Form(""),
    urgent: bool = Form(False),
    audio_file: UploadFile = File(..., alias="audio"),
):
    name, phrase = name.strip(), phrase.strip()
    if not name or not phrase:
        raise HTTPException(400, "Both the sound name and the phrase are required.")
    if len(name) > MAX_TEXT or len(phrase) > MAX_TEXT:
        raise HTTPException(400, f"Keep the sound name and phrase under {MAX_TEXT} characters.")
    wav = await _read(audio_file)
    if audio.is_silent(wav) and role != "background":  # a quiet room is a valid background take
        raise HTTPException(400, "Tink didn't hear a sound in that take. Move closer to the mic and try again.")
    takes = store.add_take(name, phrase, wav, _cues(cues), role if role in ROLES else "", urgent)
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
    top = ranked[0]
    matched = top["name"] != bayes.UNKNOWN and top["name"] in phrase
    base = {"p": top["p"], "taught": phrase.get(top["name"]), "ranked": ranked, "scores": scores, "packet": packet}

    global pending
    # The room itself: stay quiet and don't disturb an open question.
    if matched and store.role_of(top["name"]) == "background":
        return {**base, "matched": False, "name": None, "say": "", "ask": False, "source": "background",
                "background": True, "taught": None, "dialog": None}

    # Urgent sounds skip the LLM and the questions, and alert even when Tink is less sure.
    urgent = next((r for r in ranked if store.is_urgent(r["name"]) and r["p"] >= URGENT_MIN), None)
    if urgent and urgent["name"] in phrase:
        pending = None
        recent.append({"time": packet["time"], "said": phrase[urgent["name"]]})
        return {**base, "p": urgent["p"], "matched": True, "name": urgent["name"], "say": phrase[urgent["name"]],
                "ask": False, "source": "urgent", "urgent": True, "taught": phrase[urgent["name"]], "dialog": None}

    # An open yes/no question, answered with the person's own yes or no sound.
    role = store.role_of(top["name"]) if matched and top["p"] >= ANSWER_MIN else None
    if pending and role and (now - pending["at"]).total_seconds() < DIALOG_SECONDS:
        asked = pending["options"][pending["i"]]
        if role == "yes" and asked in phrase:
            pending = None
            recent.append({"time": packet["time"], "said": phrase[asked]})
            return {**base, "matched": True, "name": asked, "say": phrase[asked], "ask": False, "source": "dialog",
                    "taught": phrase[asked], "dialog": {"answered": "yes", "meant": asked}}
        pending["i"] += 1
        pending["at"] = now
        nxt = pending["options"][pending["i"]] if pending["i"] < len(pending["options"]) else None
        if nxt in phrase:
            return {**base, "matched": False, "name": None, "say": f'Do you mean "{phrase[nxt]}"?', "ask": True,
                    "source": "dialog", "dialog": {"answered": "no", "asking": nxt}}
        pending = None
        return {**base, "matched": False, "name": None, "say": "Okay. Make the sound again when you're ready.",
                "ask": True, "source": "dialog", "dialog": {"answered": "no"}}

    out = await llm.compose(packet)
    source = "grok" if out else "phrase"
    out = out or _fallback(ranked, phrase)
    if not matched or top["p"] < ASK_BELOW:
        out["ask"] = True  # never let wording turn an unsure guess into a statement

    dialog = None
    pending = None
    if out["ask"] and store.has_yes_no():
        options = [r["name"] for r in ranked if r["name"] in phrase and not store.role_of(r["name"])][:3]
        if options:
            pending = {"options": options, "i": 0, "at": now}
            out = {"say": f'Do you mean "{phrase[options[0]]}"?', "ask": True}
            source, dialog = "dialog", {"asking": options[0]}
    if not out["ask"]:
        recent.append({"time": packet["time"], "said": out["say"]})
    return {**base, "matched": matched, "name": top["name"] if matched else None,
            "say": out["say"], "ask": out["ask"], "source": source, "dialog": dialog}


@app.post("/api/reply")
async def reply(text: str = Form(...), to: str = Form("English")):
    """Caregiver answer-back: translate what they typed into the person's language and say it."""
    text, to = text.strip(), (to.strip()[:30] or "English")
    if not text:
        raise HTTPException(400, "Type what you want to say first.")
    if len(text) > 200:
        raise HTTPException(400, "Keep it under 200 characters so it is easy to hear.")
    said = await llm.translate(text, to)
    return {"say": said or text, "typed": text, "to": to, "source": "grok" if said else "typed",
            "packet": {"text": text, "to": to} if llm.KEY else None}


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
