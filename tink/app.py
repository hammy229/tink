"""Tink API. Run: uv run uvicorn tink.app:app --reload  then open http://localhost:8000

Each turn: mic clip -> voc2vec scores -> Bayesian posterior (time of day, webcam cues) -> tink.turn decides.

Every request belongs to one profile (tink/profiles.py), sent as the X-Tink-Profile header or ?p=.
Locally there is one profile and no header is needed; TINK_PUBLIC=1 turns on profiles for a hosted demo.

Concurrency: model calls and disk writes run on worker threads so the event loop stays free, and each profile
has a lock, so one person's turns run in order while other people's keep going.
"""
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import audio, bayes, llm, schemas, turn
from .embed import MODEL_ID
from .limits import RateLimiter, client_key
from .profiles import MAX_SOUNDS, MAX_TAKES, PUBLIC, Profile, Profiles
from .store import MAX_TEXT, MIN_SOUNDS, ROLES

log = logging.getLogger("tink.turn")
profiles = Profiles()
MAX_UPLOAD = 2 * 1024 * 1024  # bytes; a 4 s clip at 48 kHz is about 400 KB
RATE = 90  # requests per minute per client, public mode only
TRUST_PROXY = os.environ.get("TINK_TRUST_PROXY") == "1"  # set only behind a proxy you run (docker-compose does)
_limiter = RateLimiter(RATE)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not PUBLIC:
        await run_in_threadpool(profiles.get, None)  # load the model and the local takes before the first request
    yield


app = FastAPI(title="Tink", summary="Few-shot recognition of one person's non-word sounds.", lifespan=lifespan)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if PUBLIC and request.url.path.startswith("/api/"):
        peer = request.client.host if request.client else None
        key = client_key(request.headers.get("x-forwarded-for"), peer, TRUST_PROXY)
        if not _limiter.allow(key):
            return JSONResponse({"detail": "Too many requests. Wait a minute and try again."}, status_code=429)
    return await call_next(request)


def profile(x_tink_profile: str | None = Header(None), p: str | None = Query(None)) -> Profile:
    # A plain def, so FastAPI runs it on a worker thread: a first load can embed many takes.
    try:
        return profiles.get(x_tink_profile or p)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0])) from e


async def _read(upload: UploadFile):
    data = await upload.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That recording is too long. Keep each sound under a few seconds.")
    try:
        return await run_in_threadpool(audio.load_upload, data)
    except audio.ClipTooLong as e:
        raise HTTPException(413, "That recording is too long. Keep each sound under a few seconds.") from e
    except Exception as e:
        raise HTTPException(400, f"Could not read audio: {e}") from e


def _cues(raw: str) -> list[str]:
    try:
        out = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [c for c in out if c in bayes.CUES] if isinstance(out, list) else []


@app.get("/healthz", response_model=schemas.Health)
async def healthz():
    """Liveness. Never touches the model, so it answers even mid-inference."""
    return {"status": "ok", "model": MODEL_ID, "profiles_loaded": len(profiles.loaded)}


@app.post("/api/profiles")
async def create_profile(name: str = Form(...)):
    name = name.strip()[:MAX_TEXT]
    if not name:
        raise HTTPException(400, "Give the profile a name, like the person's first name.")
    try:
        prof = await run_in_threadpool(profiles.create, name)
    except OverflowError as e:
        raise HTTPException(503, str(e)) from e
    return {"id": prof.id, "name": prof.name}


@app.get("/api/profile")
async def whoami(prof: Profile = Depends(profile)):
    return {"id": prof.id, "name": prof.name, "public": PUBLIC}


@app.get("/api/sounds", response_model=schemas.SoundsResponse)
async def sounds(prof: Profile = Depends(profile)):
    store = prof.store
    async with prof.lock:
        listed = store.all()
    return {
        "sounds": listed,
        "threshold": store.threshold,
        "calibration": store.calib,
        "min_sounds": MIN_SOUNDS,
        "cues": bayes.CUES,
        "llm": {"on": bool(llm.KEY), "model": llm.MODEL},
        "profile": {"id": prof.id, "name": prof.name, "public": PUBLIC},
    }


@app.post("/api/enroll", response_model=schemas.EnrollResponse)
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
    async with prof.lock:
        takes = await run_in_threadpool(store.add_take, name, phrase, wav, _cues(cues), role if role in ROLES else "",
                                        urgent)
    return {"name": name, "takes": takes}


@app.post("/api/recognize", response_model=schemas.RecognizeResponse, response_model_exclude_unset=True)
async def recognize(
    cues: str = Form("[]"),
    speak_in: str = Form("English"),
    audio_file: UploadFile = File(..., alias="audio"),
    prof: Profile = Depends(profile),
):
    wav = await _read(audio_file)
    async with prof.lock:
        return await _take_turn(prof, wav, _cues(cues), speak_in)


async def _take_turn(prof: Profile, wav, cues: list[str], speak_in: str) -> dict:
    store = prof.store
    t0 = time.perf_counter()
    sounds = turn.Sounds(
        phrase={n: s["phrase"] for n, s in store.sounds.items()},
        roles={n: s["role"] for n, s in store.sounds.items() if s.get("role")},
        urgent=frozenset(n for n in store.sounds if store.is_urgent(n)),
    )
    scores = await run_in_threadpool(store.scores, wav)
    if not scores:
        raise HTTPException(409, f"Teach Tink at least {MIN_SOUNDS} sounds first, so it can tell them apart.")
    now = datetime.now()
    post = bayes.posterior(scores, store.threshold, store.usage, store.cues(), cues, now.hour,
                           temp=store.temp or bayes.TEMP)
    ranked = post["ranked"]
    heard_ms = round((time.perf_counter() - t0) * 1000)
    timing: dict = {"heard_ms": heard_ms}

    packet = {
        "time": now.strftime("%H:%M"),
        "part_of_day": post["part_of_day"],
        "speak_in": speak_in.strip()[:30] or "English",
        "candidates": [
            {"sound": r["name"], "means": sounds.phrase.get(r["name"], "(no match)"), "p": r["p"]} for r in ranked[:3]
        ],
        "webcam_cues": post["cues"],
        "recent": list(prof.recent),
    }
    outcome = turn.decide_without_wording(ranked, sounds, prof.pending, now)
    if outcome is None:
        t1 = time.perf_counter()
        worded = await llm.compose(packet)
        timing["llm_ms"] = round((time.perf_counter() - t1) * 1000) if llm.KEY else None
        outcome = turn.finish_turn(ranked, sounds, worded, now)

    prof.pending = outcome.pending
    if outcome.said:
        prof.recent.append({"time": packet["time"], "said": outcome.said})
    top = ranked[0]
    result = {
        "p": top["p"], "taught": sounds.phrase.get(top["name"]), "ranked": ranked, "scores": scores, "packet": packet,
        "timing": timing, "voiced": sorted(n for n, s in store.sounds.items() if s.get("voice")), "model": MODEL_ID,
        **outcome.fields,
    }
    # Numbers only: a person's phrases and sound names say too much to sit in a log file.
    log.info(json.dumps({"event": "turn", "source": result["source"], "matched": result["matched"], "p": result["p"],
                         "sounds": len(sounds.phrase), "heard_ms": heard_ms, "model": MODEL_ID}))
    return result


@app.post("/api/reply", response_model=schemas.ReplyResponse)
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
    async with prof.lock:
        banked = await run_in_threadpool(prof.store.set_voice, name, wav)
    if not banked:
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
    async with prof.lock:
        confirmed = await run_in_threadpool(prof.store.confirm, name, datetime.now().hour)
    if not confirmed:
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True, "usage": prof.store.usage.get(name, {})}


@app.delete("/api/sounds/{name:path}")
async def forget(name: str, prof: Profile = Depends(profile)):
    async with prof.lock:
        deleted = await run_in_threadpool(prof.store.delete, name)
    if not deleted:
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")
