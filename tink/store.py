"""Enrolled sounds on disk, plus centered-cosine scoring.

Layout:
    data/sounds.json             {"Low hum": {"phrase": "I'm thirsty.", "cues": ["smile"], "takes": ["low-hum/1.wav"]}}
    data/usage.json              {"Low hum": {"afternoon": 3}}   confirmed uses per part of day (Bayesian prior)
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
from .bayes import CUES, part_of_day
from .embed import embed

DATA = Path(os.environ.get("TINK_DATA", "data"))
THRESHOLD = float(os.environ.get("TINK_THRESHOLD", "0.5"))  # on centered cosine; calibrate with risk_test.py


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "sound"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


class Store:
    def __init__(self, root: Path = DATA):
        self.root = root
        self.sounds: dict[str, dict] = _read(root / "sounds.json")
        self.usage: dict[str, dict[str, int]] = _read(root / "usage.json")
        self.vecs: dict[str, list[np.ndarray]] = {
            name: [embed(audio.load(root / "takes" / t)) for t in s["takes"]] for name, s in self.sounds.items()
        }

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sounds.json").write_text(json.dumps(self.sounds, indent=2))
        (self.root / "usage.json").write_text(json.dumps(self.usage, indent=2))

    def all(self) -> list[dict]:
        return [
            {"name": n, "phrase": s["phrase"], "cues": s.get("cues", []), "takes": len(s["takes"])}
            for n, s in self.sounds.items()
        ]

    def cues(self) -> dict[str, list[str]]:
        return {n: s.get("cues", []) for n, s in self.sounds.items()}

    def add_take(self, name: str, phrase: str, wav: np.ndarray, cues: list[str] | None = None) -> int:
        s = self.sounds.setdefault(name, {"phrase": phrase, "cues": [], "takes": []})
        s["phrase"] = phrase
        if cues is not None:
            s["cues"] = [c for c in cues if c in CUES]
        rel = f"{slug(name)}/{len(s['takes']) + 1}.wav"
        path = self.root / "takes" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio.to_wav_bytes(wav))
        s["takes"].append(rel)
        self.vecs.setdefault(name, []).append(embed(wav))
        self._save()
        return len(s["takes"])

    def confirm(self, name: str, hour: int) -> bool:
        if name not in self.sounds:
            return False
        b = self.usage.setdefault(name, {})
        b[part_of_day(hour)] = b.get(part_of_day(hour), 0) + 1
        self._save()
        return True

    def delete(self, name: str) -> bool:
        if name not in self.sounds:
            return False
        shutil.rmtree(self.root / "takes" / slug(name), ignore_errors=True)
        del self.sounds[name], self.vecs[name]
        self.usage.pop(name, None)
        self._save()
        return True

    def scores(self, wav: np.ndarray) -> dict[str, float]:
        """Centered cosine between the clip and each sound's averaged takes."""
        names = [n for n, v in self.vecs.items() if v]
        if not names:
            return {}
        # Raw voc2vec means are all very similar, so cosine on raw vectors bunches near 1.
        # Subtracting the mean of all takes removes that shared component and spreads them out.
        everything = np.concatenate([np.stack(self.vecs[n]) for n in names])
        mu = everything.mean(axis=0) if len(names) > 1 else 0
        cents = _unit(np.stack([np.stack(self.vecs[n]).mean(axis=0) - mu for n in names]))
        s = cents @ _unit(embed(wav) - mu)
        return {n: round(float(v), 3) for n, v in zip(names, s)}
