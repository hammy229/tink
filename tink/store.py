"""Enrolled sounds on disk, plus nearest-centroid matching.

Layout:
    data/sounds.json             {"Low hum": {"phrase": "I'm thirsty.", "takes": ["low-hum/1.wav", ...]}}
    data/takes/<slug>/<n>.wav    16 kHz mono clips
Embeddings are recomputed from the clips at startup, so changing TINK_LAYER needs no migration.
"""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from . import audio
from .embed import embed

DATA = Path(os.environ.get("TINK_DATA", "data"))
THRESHOLD = float(os.environ.get("TINK_THRESHOLD", "0.5"))  # on centered cosine; calibrate with risk_test.py


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "sound"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Store:
    def __init__(self, root: Path = DATA):
        self.root = root
        self.index = root / "sounds.json"
        self.sounds: dict[str, dict] = json.loads(self.index.read_text()) if self.index.exists() else {}
        self.vecs: dict[str, list[np.ndarray]] = {
            name: [embed(audio.load(root / "takes" / t)) for t in s["takes"]] for name, s in self.sounds.items()
        }

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.index.write_text(json.dumps(self.sounds, indent=2))

    def list(self) -> list[dict]:
        return [{"name": n, "phrase": s["phrase"], "takes": len(s["takes"])} for n, s in self.sounds.items()]

    def add_take(self, name: str, phrase: str, wav: np.ndarray) -> int:
        s = self.sounds.setdefault(name, {"phrase": phrase, "takes": []})
        s["phrase"] = phrase
        rel = f"{slug(name)}/{len(s['takes']) + 1}.wav"
        path = self.root / "takes" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio.to_wav_bytes(wav))
        s["takes"].append(rel)
        self.vecs.setdefault(name, []).append(embed(wav))
        self._save()
        return len(s["takes"])

    def delete(self, name: str) -> bool:
        if name not in self.sounds:
            return False
        shutil.rmtree(self.root / "takes" / slug(name), ignore_errors=True)
        del self.sounds[name], self.vecs[name]
        self._save()
        return True

    def recognize(self, wav: np.ndarray) -> dict:
        names = [n for n, v in self.vecs.items() if v]
        if not names:
            return {"matched": False, "name": None, "phrase": None, "score": 0.0, "scores": {}}
        # Center on the mean of all takes: raw voc2vec means are all very similar, so cosine
        # on raw vectors bunches near 1. Subtracting the shared component spreads them out.
        everything = np.concatenate([np.stack(self.vecs[n]) for n in names])
        mu = everything.mean(axis=0) if len(names) > 1 else 0
        cents = _unit(np.stack([np.stack(self.vecs[n]).mean(axis=0) - mu for n in names]))
        scores = cents @ _unit(embed(wav) - mu)
        best = int(scores.argmax())
        matched = bool(scores[best] >= THRESHOLD)
        return {
            "matched": matched,
            "name": names[best] if matched else None,
            "phrase": self.sounds[names[best]]["phrase"] if matched else None,
            "score": round(float(scores[best]), 3),
            "scores": {n: round(float(s), 3) for n, s in zip(names, scores)},
        }
