"""Write what the Python code does on fixed inputs, so the browser engine can be held to the same answers.

    uv run python -m scripts.gen_parity

tests/test_vad_and_parity.py fails if this output goes stale, and web/engine/core.test.mjs fails if JS disagrees.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from scipy.signal import resample_poly

from tink import audio, bayes, turn
from tink.store import Store
from tink.vad import SegmentDetector

OUT = Path(__file__).parent.parent / "tests" / "fixtures" / "parity.json"
NOW = datetime(2026, 11, 5, 14, 0)


def _floats(a) -> list:
    return np.round(np.asarray(a, dtype=float), 6).tolist()


def _vectors() -> dict:
    rng = np.random.default_rng(42)
    centers = {name: rng.normal(size=8) for name in ("Hum", "Hiss", "Click")}
    return {
        name: [[(c + rng.normal(0, 0.7, 8)).astype(np.float32) for _ in range(2)] for _ in range(3)]
        for name, c in centers.items()
    }


def _scoring() -> dict:
    vecs = _vectors()
    query = np.random.default_rng(7).normal(size=8).astype(np.float32)
    with TemporaryDirectory() as tmp, mock.patch("tink.store.embed", lambda _: query):
        store = Store(Path(tmp))
        store.vecs = {n: [np.stack(t) for t in takes] for n, takes in vecs.items()}
        calib = store.calibrate()
        scores = store.scores(np.zeros(16_000, np.float32))
    return {
        "vecs": {n: [[_floats(r) for r in t] for t in takes] for n, takes in vecs.items()},
        "query": _floats(query),
        "scores": scores,
        "calibrate": calib,
    }


def _posteriors() -> list:
    cases = [
        ({"Hum": 0.7, "Sigh": 0.2}, 0.5, {}, {}, [], 14, 0.1),
        ({"Hum": 0.1, "Sigh": 0.05}, 0.5, {}, {}, [], 2, 0.1),
        ({"Hum": 0.55, "Sigh": 0.53}, 0.5, {"Sigh": {"afternoon": 6}}, {}, [], 14, 0.1),
        ({"Hum": 0.55, "Sigh": 0.53}, 0.4, {}, {"Hum": ["smile"], "Sigh": ["eyes_closed"]}, ["eyes_closed", "bad"], 20,
         0.3),
    ]
    return [{"args": list(c), "out": bayes.posterior(*c[:6], temp=c[6])} for c in cases]


def _ranked(*pairs) -> list[dict]:
    return [{"name": n, "p": p, "audio_p": p, "prior": 0.1} for n, p in pairs]


def _pending(p: turn.Pending | None) -> dict | None:
    if p is None:
        return None
    return {"options": list(p.options), "i": p.i, "at_seconds": (p.at - NOW).total_seconds()}


def _turns() -> list:
    sounds = turn.Sounds(
        {"Hum": "I'm thirsty.", "Hiss": "I'm cold.", "Ah": "Yes", "Mm": "No", "Room": "(room)", "Cry": "It hurts!"},
        {"Ah": "yes", "Mm": "no", "Room": "background"},
        frozenset({"Cry"}),
    )
    bare = turn.Sounds({"Hum": "a", "Hiss": "b"}, {}, frozenset())
    open_q = turn.Pending(("Hum", "Hiss"), 0, NOW - timedelta(seconds=5))
    last_q = turn.Pending(("Hum", "Hiss"), 1, NOW - timedelta(seconds=5))
    stale = turn.Pending(("Hum",), 0, NOW - timedelta(seconds=46))
    cases = [
        (_ranked(("Room", 0.9), ("Hum", 0.1)), sounds, open_q),
        (_ranked(("Hum", 0.5), ("Cry", 0.45)), sounds, None),
        (_ranked(("Hum", 0.7), ("Cry", 0.2)), sounds, None),
        (_ranked(("Ah", 0.9), ("Hum", 0.1)), sounds, open_q),
        (_ranked(("Mm", 0.9), ("Hum", 0.1)), sounds, open_q),
        (_ranked(("Mm", 0.9), ("Hum", 0.1)), sounds, last_q),
        (_ranked(("Ah", 0.5), ("Hum", 0.3)), sounds, open_q),
        (_ranked(("Ah", 0.9), ("Hum", 0.1)), sounds, stale),
        (_ranked(("Hum", 0.9), ("Hiss", 0.1)), sounds, None),
        (_ranked(("Hum", 0.4), ("Ah", 0.3), ("Hiss", 0.2), ("unknown", 0.1)), sounds, None),
        (_ranked(("Hum", 0.5), ("Hiss", 0.4)), bare, None),
        (_ranked(("unknown", 0.8), ("Hum", 0.2)), bare, None),
    ]
    out = []
    for ranked, s, pending in cases:
        o = turn.decide_without_wording(ranked, s, pending, NOW) or turn.finish_turn(ranked, s, None, NOW)
        out.append({
            "ranked": ranked,
            "sounds": {"phrase": s.phrase, "roles": s.roles, "urgent": sorted(s.urgent)},
            "pending": _pending(pending),
            "fields": o.fields,
            "next_pending": _pending(o.pending),
            "said": o.said,
        })
    return out


def _audio() -> dict:
    rng = np.random.default_rng(3)
    t = np.arange(4000) / 16_000
    wav = (0.4 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.01, 4000)).astype(np.float32)
    padded = np.concatenate([np.zeros(6000, np.float32), wav, np.zeros(6000, np.float32)])
    return {
        "wav": _floats(wav[:800]),
        "resample": {f"{u}/{d}": _floats(resample_poly(wav[:800], u, d)) for u, d in ((16, 17), (17, 16), (1, 3))},
        "trim_len": len(audio.trim(padded)),
        "trim_input_len": len(padded),
        "short_trim_len": len(audio.trim(np.full(100, 0.5, np.float32))),
        "silent": bool(audio.is_silent(np.zeros(16_000, np.float32))),
    }


def _vad_stream() -> tuple[list[tuple[str, int]], list[int]]:
    """A scripted stream: quiet room, sound, quiet, a tick too short to count, louder room, sound."""
    plan = [("quiet", 40), ("sound", 40), ("quiet", 40), ("tick", 3), ("quiet", 40), ("fan", 150), ("fan+sound", 40),
            ("fan", 60), ("long", 260), ("quiet", 40), ("sound", 8)]
    levels = {"quiet": 0.001, "sound": 0.2, "tick": 0.2, "fan": 0.02, "fan+sound": 0.3, "long": 0.2}
    frames = []
    for kind, n in plan:
        for i in range(n):
            frames.append(np.full(320, levels[kind] * (1 if i % 2 else -1), np.float32))
    stream = np.concatenate(frames)
    det = SegmentDetector()
    lengths = [len(s) for chunk in np.array_split(stream, 37) for s in det.push(chunk)]
    lengths += [len(s) for s in det.flush()]
    return plan, lengths


def build() -> dict:
    plan, lengths = _vad_stream()
    return {"scoring": _scoring(), "posterior": _posteriors(), "turn": _turns(), "audio": _audio(),
            "vad": {"plan": plan, "segment_lengths": lengths}}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True, parents=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")
