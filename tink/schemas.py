"""Response contracts. FastAPI validates against these and publishes them at /docs."""
from typing import Any

from pydantic import BaseModel


class RankedSound(BaseModel):
    name: str
    p: float
    audio_p: float
    prior: float


class Timing(BaseModel):
    heard_ms: int
    llm_ms: int | None = None


class RecognizeResponse(BaseModel):
    matched: bool
    name: str | None
    say: str
    ask: bool
    source: str  # phrase, grok, dialog, urgent, or background
    p: float
    taught: str | None
    ranked: list[RankedSound]
    scores: dict[str, float]
    packet: dict[str, Any]
    timing: Timing
    voiced: list[str]
    model: str
    dialog: dict[str, Any] | None = None
    urgent: bool = False
    background: bool = False


class SoundInfo(BaseModel):
    name: str
    phrase: str
    cues: list[str]
    role: str | None
    urgent: bool
    voice: bool
    takes: int


class ProfileInfo(BaseModel):
    id: str
    name: str
    public: bool


class LlmInfo(BaseModel):
    on: bool
    model: str


class SoundsResponse(BaseModel):
    sounds: list[SoundInfo]
    threshold: float
    calibration: dict[str, Any]
    min_sounds: int
    cues: tuple[str, ...]
    llm: LlmInfo
    profile: ProfileInfo


class EnrollResponse(BaseModel):
    name: str
    takes: int


class ReplyResponse(BaseModel):
    say: str
    typed: str
    to: str
    source: str
    packet: dict[str, Any] | None


class Health(BaseModel):
    status: str
    model: str
    profiles_loaded: int
