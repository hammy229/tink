"""Tink API. Run: uv run uvicorn tink.app:app --reload  then open http://localhost:8000

Each turn: mic clip -> voc2vec scores -> Bayesian posterior (time of day, webcam cues)
-> JSON packet -> Grok -> what Tink says, in the listener's language.

Every request belongs to one profile (tink/profiles.py), sent as the X-Tink-Profile header or ?p=.
Locally there is one profile and no header is needed; TINK_PUBLIC=1 turns on profiles for a hosted demo.

Routes are `async def` on purpose: apart from the LLM call they never await while touching a Store,
so on uvicorn's single event loop they run one at a time and can't interleave mid-mutation.
"""
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import audio, bayes, llm
from .profiles import MAX_SOUNDS, MAX_TAKES, PUBLIC, Profile, Profiles
from .store import MAX_TEXT, MIN_SOUNDS, ROLES

app = FastAPI(title="Tink")
profiles = Profiles()
if not PUBLIC:
    profiles.get(None)  # load the model and the local takes at startup
ASK_BELOW = 0.6  # top probability under this means Tink asks instead of saying
DIALOG_SECONDS = 45  # how long a yes/no question stays open
ANSWER_MIN = 0.6  # a yes or no sound must be at least this sure to answer a question
URGENT_MIN = 0.4  # urgent sounds alert at lower confidence: a missed alarm costs more than a false one
MAX_UPLOAD = 2 * 1024 * 1024  # bytes; a 4 s clip at 48 kHz is about 400 KB
RATE = 90  # requests per minute per client, public mode only
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if PUBLIC and request.url.path.startswith("/api/"):
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
        now, q = time.monotonic(), _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE:
            return JSONResponse({"detail": "Too many requests. Wait a minute and try again."}, status_code=429)
        q.append(now)
    return await call_next(request)


def profile(x_tink_profile: str | None = Header(None), p: str | None = Query(None)) -> Profile:
    try:
        return profiles.get(x_tink_profile or p)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0])) from e


async def _read(upload: UploadFile):
    data = await upload.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That recording is too long. Keep each sound under a few seconds.")
    try:
        return audio.load(data)
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


@app.post("/api/profiles")
async def create_profile(name: str = Form(...)):
    name = name.strip()[:MAX_TEXT]
    if not name:
        raise HTTPException(400, "Give the profile a name, like the person's first name.")
    try:
        prof = profiles.create(name)
    except OverflowError as e:
        raise HTTPException(503, str(e)) from e
    return {"id": prof.id, "name": prof.name}


@app.get("/api/profile")
async def whoami(prof: Profile = Depends(profile)):
    return {"id": prof.id, "name": prof.name, "public": PUBLIC}


@app.get("/api/sounds")
async def sounds(prof: Profile = Depends(profile)):
    store = prof.store
    return {
        "sounds": store.all(),
        "threshold": store.threshold,
        "calibration": store.calib,
        "min_sounds": MIN_SOUNDS,
        "cues": bayes.CUES,
        "llm": {"on": bool(llm.KEY), "model": llm.MODEL},
        "profile": {"id": prof.id, "name": prof.name, "public": PUBLIC},
    }


@app.post("/api/enroll")
async def enroll(
    name: str = Form(...),
    phrase: str = Form(...),
    cues: str = Form("[]"),
    role: str = Form(""),
    urgent: bool = Form(False),
    audio_file: UploadFile = File(..., alias="audio"),
    prof: Profile = Depends(profile),
):
    store = prof.store
    name, phrase = name.strip(), phrase.strip()
    if not name or not phrase:
        raise HTTPException(400, "Both the sound name and the phrase are required.")
    if len(name) > MAX_TEXT or len(phrase) > MAX_TEXT:
        raise HTTPException(400, f"Keep the sound name and phrase under {MAX_TEXT} characters.")
    if PUBLIC and name not in store.sounds and len(store.sounds) >= MAX_SOUNDS:
        raise HTTPException(409, f"This demo keeps up to {MAX_SOUNDS} sounds per person. Forget one to add another.")
    if PUBLIC and len(store.sounds.get(name, {}).get("takes", [])) >= MAX_TAKES:
        raise HTTPException(409, f"That sound already has {MAX_TAKES} takes, which is plenty.")
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
    prof: Profile = Depends(profile),
):
    store = prof.store
    t0 = time.perf_counter()
    wav = await _read(audio_file)
    phrase = {n: s["phrase"] for n, s in store.sounds.items()}  # snapshot: a Forget mid-request can't break this turn
    scores = store.scores(wav)
    if not scores:
        raise HTTPException(409, f"Teach Tink at least {MIN_SOUNDS} sounds first, so it can tell them apart.")
    now = datetime.now()
    post = bayes.posterior(scores, store.threshold, store.usage, store.cues(), _cues(cues), now.hour,
                           temp=store.temp or bayes.TEMP)
    ranked = post["ranked"]
    heard_ms = round((time.perf_counter() - t0) * 1000)

    packet = {
        "time": now.strftime("%H:%M"),
        "part_of_day": post["part_of_day"],
        "speak_in": speak_in.strip()[:30] or "English",
        "candidates": [
            {"sound": r["name"], "means": phrase.get(r["name"], "(no match)"), "p": r["p"]} for r in ranked[:3]
        ],
        "webcam_cues": post["cues"],
        "recent": list(prof.recent),
    }
    top = ranked[0]
    matched = top["name"] != bayes.UNKNOWN and top["name"] in phrase
    voiced = {n for n, s in store.sounds.items() if s.get("voice")}
    base = {"p": top["p"], "taught": phrase.get(top["name"]), "ranked": ranked, "scores": scores, "packet": packet,
            "timing": {"heard_ms": heard_ms}, "voiced": sorted(voiced)}

    # The room itself: stay quiet and don't disturb an open question.
    if matched and store.role_of(top["name"]) == "background":
        return {**base, "matched": False, "name": None, "say": "", "ask": False, "source": "background",
                "background": True, "taught": None, "dialog": None}

    # Urgent sounds skip the LLM and the questions, and alert even when Tink is less sure.
    urgent = next((r for r in ranked if store.is_urgent(r["name"]) and r["p"] >= URGENT_MIN), None)
    if urgent and urgent["name"] in phrase:
        prof.pending = None
        prof.recent.append({"time": packet["time"], "said": phrase[urgent["name"]]})
        return {**base, "p": urgent["p"], "matched": True, "name": urgent["name"], "say": phrase[urgent["name"]],
                "ask": False, "source": "urgent", "urgent": True, "taught": phrase[urgent["name"]], "dialog": None}

    # An open yes/no question, answered with the person's own yes or no sound.
    pending = prof.pending
    role = store.role_of(top["name"]) if matched and top["p"] >= ANSWER_MIN else None
    if pending and role and (now - pending["at"]).total_seconds() < DIALOG_SECONDS:
        asked = pending["options"][pending["i"]]
        if role == "yes" and asked in phrase:
            prof.pending = None
            prof.recent.append({"time": packet["time"], "said": phrase[asked]})
            return {**base, "matched": True, "name": asked, "say": phrase[asked], "ask": False, "source": "dialog",
                    "taught": phrase[asked], "dialog": {"answered": "yes", "meant": asked}}
        pending["i"] += 1
        pending["at"] = now
        nxt = pending["options"][pending["i"]] if pending["i"] < len(pending["options"]) else None
        if nxt in phrase:
            return {**base, "matched": False, "name": None, "say": f'Do you mean "{phrase[nxt]}"?', "ask": True,
                    "source": "dialog", "dialog": {"answered": "no", "asking": nxt}}
        prof.pending = None
        return {**base, "matched": False, "name": None, "say": "Okay. Make the sound again when you're ready.",
                "ask": True, "source": "dialog", "dialog": {"answered": "no"}}

    t1 = time.perf_counter()
    out = await llm.compose(packet)
    base["timing"]["llm_ms"] = round((time.perf_counter() - t1) * 1000) if llm.KEY else None
    source = "grok" if out else "phrase"
    out = out or _fallback(ranked, phrase)
    if not matched or top["p"] < ASK_BELOW:
        out["ask"] = True  # never let wording turn an unsure guess into a statement

    dialog = None
    prof.pending = None
    if out["ask"] and store.has_yes_no():
        options = [r["name"] for r in ranked if r["name"] in phrase and not store.role_of(r["name"])][:3]
        if options:
            prof.pending = {"options": options, "i": 0, "at": now}
            out = {"say": f'Do you mean "{phrase[options[0]]}"?', "ask": True}
            source, dialog = "dialog", {"asking": options[0]}
    if not out["ask"]:
        prof.recent.append({"time": packet["time"], "said": out["say"]})
    return {**base, "matched": matched, "name": top["name"] if matched else None,
            "say": out["say"], "ask": out["ask"], "source": source, "dialog": dialog}


@app.post("/api/reply")
async def reply(text: str = Form(...), to: str = Form("English"), prof: Profile = Depends(profile)):
    """Caregiver answer-back: translate what they typed into the person's language and say it."""
    text, to = text.strip(), (to.strip()[:30] or "English")
    if not text:
        raise HTTPException(400, "Type what you want to say first.")
    if len(text) > 200:
        raise HTTPException(400, "Keep it under 200 characters so it is easy to hear.")
    said = await llm.translate(text, to)
    return {"say": said or text, "typed": text, "to": to, "source": "grok" if said else "typed",
            "packet": {"text": text, "to": to} if llm.KEY else None}


@app.post("/api/voice")
async def bank_voice(name: str = Form(...), audio_file: UploadFile = File(..., alias="audio"),
                     prof: Profile = Depends(profile)):
    """Message banking: store the person saying this sound's phrase in their own voice."""
    wav = await _read(audio_file)
    if audio.is_silent(wav):
        raise HTTPException(400, "Tink didn't hear the phrase. Move closer to the mic and try again.")
    if not prof.store.set_voice(name, wav):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True, "name": name}


@app.get("/api/voice/{name:path}")
async def play_voice(name: str, prof: Profile = Depends(profile)):
    path = prof.store.voice_path(name)
    if not path:
        raise HTTPException(404, f"No banked voice for {name!r}.")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/confirm")
async def confirm(name: str = Form(...), prof: Profile = Depends(profile)):
    """Someone pressed "Right": teach the time-of-day prior that this sound gets used now."""
    if not prof.store.confirm(name, datetime.now().hour):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True, "usage": prof.store.usage.get(name, {})}


@app.delete("/api/sounds/{name:path}")
async def forget(name: str, prof: Profile = Depends(profile)):
    if not prof.store.delete(name):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")
