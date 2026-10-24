"""Enrolled sounds on disk, plus centered-cosine scoring.

Layout:
    data/sounds.json             {"Low hum": {"phrase": "I'm thirsty.", "cues": ["smile"], "dir": "low-hum-3f9a2c",
                                              "takes": ["low-hum-3f9a2c/8b1e04d2.wav"]}}
    data/usage.json              {"Low hum": {"afternoon": 3}}   confirmed uses per part of day (Bayesian prior)
    data/takes/<dir>/<id>.wav    16 kHz mono clips; <dir> is unique per sound so similar names never collide
Embeddings are recomputed from the clips at startup, so changing TINK_LAYER needs no migration.

Each take is stored as a small matrix: row 0 is the take itself, the other rows are its augmented variants
(audio.augment). A sound's centroid averages every row of every take.

After every change, calibrate() runs a self-test: each original take is held out (with its own variants,
so it can't recognise itself) and matched against the rest. That fits the softmax temperature and the
"didn't catch it" threshold to this person's sounds, and reports how often Tink was right.
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
THRESHOLD = float(os.environ.get("TINK_THRESHOLD", "0.5"))  # fallback until calibration has enough takes
AUGMENT = os.environ.get("TINK_AUGMENT", "1") != "0"
TEMPS = np.geomspace(0.04, 1.0, 50)  # floor: a handful of easy takes must not make Tink certain
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
        self.vecs: dict[str, list[np.ndarray]] = {}  # per take: (1 + variants, dim)
        for name, s in self.sounds.items():
            s["takes"] = [t for t in s["takes"] if (root / "takes" / t).exists()]  # survive a lost file
            self.vecs[name] = [self._take_vecs(audio.load(root / "takes" / t)) for t in s["takes"]]
        if not self.vecs:
            embed(np.zeros(audio.SR // 2, np.float32))  # load the model now, not on the first request
        self.calib: dict = {}
        self.calibrate()

    @staticmethod
    def _take_vecs(wav: np.ndarray) -> np.ndarray:
        variants = audio.augment(wav) if AUGMENT else []
        return np.stack([embed(w) for w in [wav, *variants]])

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sounds.json").write_text(json.dumps(self.sounds, indent=2))
        (self.root / "usage.json").write_text(json.dumps(self.usage, indent=2))

    def all(self) -> list[dict]:
        return [
            {"name": n, "phrase": s["phrase"], "cues": s.get("cues", []), "role": s.get("role"),
             "urgent": bool(s.get("urgent")), "voice": bool(s.get("voice")), "takes": len(s["takes"])}
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
        vec = self._take_vecs(wav)  # before touching state, so a model failure leaves nothing half-written
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
        self.calibrate()
        return len(s["takes"])

    # Message banking: the person records the phrase in their own voice while they still can,
    # and Tink plays that recording instead of a synthetic voice.
    def set_voice(self, name: str, wav: np.ndarray) -> bool:
        s = self.sounds.get(name)
        if not s:
            return False
        s.setdefault("dir", f"{slug(name)[:40]}-{uuid.uuid4().hex[:6]}")
        rel = f"{s['dir']}/voice.wav"
        (self.root / "takes" / s["dir"]).mkdir(parents=True, exist_ok=True)
        (self.root / "takes" / rel).write_bytes(audio.to_wav_bytes(wav))
        s["voice"] = rel
        self._save()
        return True

    def voice_path(self, name: str) -> Path | None:
        rel = self.sounds.get(name, {}).get("voice")
        path = self.root / "takes" / rel if rel else None
        return path if path and path.exists() else None

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
        self.calibrate()
        return True

    def scores(self, wav: np.ndarray) -> dict[str, float]:
        """Centered cosine between the clip and each sound's averaged takes."""
        names = [n for n, v in self.vecs.items() if v]
        if len(names) < MIN_SOUNDS:
            return {}
        # Raw voc2vec means are all very similar, so cosine on raw vectors bunches near 1.
        # Subtracting the mean of all takes removes that shared component and spreads them out.
        everything = np.concatenate([np.concatenate(self.vecs[n]) for n in names])
        mu = everything.mean(axis=0)
        cents = _unit(np.stack([np.concatenate(self.vecs[n]).mean(axis=0) - mu for n in names]))
        s = cents @ _unit(embed(wav) - mu)
        return {n: round(float(v), 3) for n, v in zip(names, s)}

    # ---------- calibration ----------
    @property
    def temp(self) -> float | None:
        return self.calib.get("temp")

    @property
    def threshold(self) -> float:
        return self.calib.get("threshold", THRESHOLD)

    def calibrate(self) -> dict:
        """Leave-one-take-out self-test; fits temperature and threshold when there is enough data."""
        names = [n for n, v in self.vecs.items() if v]
        rows, labels = [], []
        for si, n in enumerate(names):
            if len(self.vecs[n]) < 2:
                continue
            for ti, held in enumerate(self.vecs[n]):
                kept = {m: [v for j, v in enumerate(self.vecs[m]) if not (m == n and j == ti)] for m in names}
                mu = np.concatenate([np.concatenate(v) for v in kept.values()]).mean(axis=0)
                cents = _unit(np.stack([np.concatenate(kept[m]).mean(axis=0) - mu for m in names]))
                rows.append(cents @ _unit(held[0] - mu))
                labels.append(si)
        if len(rows) < 4 or len({*labels}) < 2:
            self.calib = {"n": len(rows), "ready": False}
            return self.calib
        S, y = np.stack(rows), np.array(labels)
        right = S[np.arange(len(y)), y]
        wrong = np.where(np.eye(len(names), dtype=bool)[y], -np.inf, S).max(axis=1)
        correct = S.argmax(axis=1) == y

        def nll(t):
            z = S / t
            z = z - z.max(axis=1, keepdims=True)
            return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean())

        temp = float(min(TEMPS, key=nll))
        p = np.exp((S - S.max(axis=1, keepdims=True)) / temp)
        conf = (p / p.sum(axis=1, keepdims=True)).max(axis=1)
        sure = conf >= 0.9
        threshold = float(np.clip((np.median(right) + np.median(wrong)) / 2, 0.1, 0.9))
        self.calib = {
            "ready": True, "n": int(len(y)), "right": int(correct.sum()),
            "accuracy": round(float(correct.mean()), 3),
            "sure_n": int(sure.sum()), "sure_right": int((correct & sure).sum()),
            "temp": round(temp, 4), "threshold": round(threshold, 3),
        }
        return self.calib
