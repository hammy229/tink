"""Enrolled sounds on disk, plus centered-cosine scoring.

Layout:
    data/sounds.json             {"Low hum": {"phrase": "I'm thirsty.", "cues": ["smile"], "dir": "low-hum-3f9a2c",
                                              "takes": ["low-hum-3f9a2c/8b1e04d2.wav"]}}
    data/usage.json              {"Low hum": {"afternoon": 3}}   confirmed uses per part of day (Bayesian prior)
    data/takes/<dir>/<id>.wav    16 kHz mono clips; <dir> is unique per sound so similar names never collide
Embeddings are recomputed from the clips at startup, so changing TINK_LAYER needs no migration.
"""
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import numpy as np

from . import audio
from .bayes import CUES, part_of_day
from .embed import embed

DATA = Path(os.environ.get("TINK_DATA", "data"))
THRESHOLD = float(os.environ.get("TINK_THRESHOLD", "0.5"))  # on centered cosine; calibrate with risk_test.py
MIN_SOUNDS = 2  # centering needs two sounds; with one, every clip looks like a match
MAX_TEXT = 80
ROLES = ("yes", "no", "background")  # yes/no answer Tink's questions; background is the room itself


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
        self.vecs: dict[str, list[np.ndarray]] = {}
        for name, s in self.sounds.items():
            s["takes"] = [t for t in s["takes"] if (root / "takes" / t).exists()]  # survive a lost file
            self.vecs[name] = [embed(audio.load(root / "takes" / t)) for t in s["takes"]]
        if not self.vecs:
            embed(np.zeros(audio.SR // 2, np.float32))  # load the model now, not on the first request

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sounds.json").write_text(json.dumps(self.sounds, indent=2))
        (self.root / "usage.json").write_text(json.dumps(self.usage, indent=2))

    def all(self) -> list[dict]:
        return [
            {"name": n, "phrase": s["phrase"], "cues": s.get("cues", []), "role": s.get("role"),
             "urgent": bool(s.get("urgent")), "takes": len(s["takes"])}
            for n, s in self.sounds.items()
        ]

    def cues(self) -> dict[str, list[str]]:
        return {n: s.get("cues", []) for n, s in self.sounds.items()}

    def role_of(self, name: str | None) -> str | None:
        return self.sounds.get(name, {}).get("role") if name else None

    def is_urgent(self, name: str | None) -> bool:
        return bool(self.sounds.get(name, {}).get("urgent")) if name else False

    def has_yes_no(self) -> bool:
        return {"yes", "no"} <= {s.get("role") for s in self.sounds.values()}

    def add_take(self, name: str, phrase: str, wav: np.ndarray, cues: list[str] | None = None, role: str | None = None,
                 urgent: bool | None = None) -> int:
        vec = embed(wav)  # before touching state, so a model failure leaves nothing half-written
        s = self.sounds.setdefault(name, {"phrase": phrase, "cues": [], "takes": []})
        s.setdefault("dir", f"{slug(name)[:40]}-{uuid.uuid4().hex[:6]}")
        s["phrase"] = phrase
        if cues is not None:
            s["cues"] = [c for c in cues if c in CUES]
        if role in ROLES:
            for other in self.sounds.values():  # one yes sound and one no sound at a time
                if other.get("role") == role:
                    other.pop("role")
            s["role"] = role
        elif role == "":
            s.pop("role", None)
        if urgent is not None:
            s["urgent"] = urgent and s.get("role") != "background"
        rel = f"{s['dir']}/{uuid.uuid4().hex[:8]}.wav"
        path = self.root / "takes" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio.to_wav_bytes(wav))
        s["takes"].append(rel)
        self.vecs.setdefault(name, []).append(vec)
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
        s = self.sounds.pop(name)
        for folder in {Path(t).parent for t in s["takes"]} | ({Path(s["dir"])} if "dir" in s else set()):
            shutil.rmtree(self.root / "takes" / folder, ignore_errors=True)
        self.vecs.pop(name, None)
        self.usage.pop(name, None)
        self._save()
        return True

    def scores(self, wav: np.ndarray) -> dict[str, float]:
        """Centered cosine between the clip and each sound's averaged takes."""
        names = [n for n, v in self.vecs.items() if v]
        if len(names) < MIN_SOUNDS:
            return {}
        # Raw voc2vec means are all very similar, so cosine on raw vectors bunches near 1.
        # Subtracting the mean of all takes removes that shared component and spreads them out.
        everything = np.concatenate([np.stack(self.vecs[n]) for n in names])
        mu = everything.mean(axis=0)
        cents = _unit(np.stack([np.stack(self.vecs[n]).mean(axis=0) - mu for n in names]))
        s = cents @ _unit(embed(wav) - mu)
        return {n: round(float(v), 3) for n, v in zip(names, s)}
