"""Tink API. Run: uv run uvicorn tink.app:app --reload  then open http://localhost:8000"""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import audio
from .store import THRESHOLD, Store

app = FastAPI(title="Tink")
store = Store()


async def _read(upload: UploadFile):
    try:
        return audio.load(await upload.read())
    except Exception as e:
        raise HTTPException(400, f"Could not read audio: {e}") from e


@app.get("/api/sounds")
def sounds():
    return {"sounds": store.list(), "threshold": THRESHOLD}


@app.post("/api/enroll")
async def enroll(name: str = Form(...), phrase: str = Form(...), audio_file: UploadFile = File(..., alias="audio")):
    name, phrase = name.strip(), phrase.strip()
    if not name or not phrase:
        raise HTTPException(400, "Both the sound name and the phrase are required.")
    takes = store.add_take(name, phrase, await _read(audio_file))
    return {"name": name, "takes": takes}


@app.post("/api/recognize")
async def recognize(audio_file: UploadFile = File(..., alias="audio")):
    return store.recognize(await _read(audio_file))


@app.delete("/api/sounds/{name}")
def forget(name: str):
    if not store.delete(name):
        raise HTTPException(404, f"No sound called {name!r}.")
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="web")
