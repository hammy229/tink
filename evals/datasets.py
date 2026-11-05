"""Data contracts for the benchmarks. One clip = one audio file from one person in one session.

ReCANVo (Narain and Johnson 2021, doi:10.5281/zenodo.5786860, CC BY 3.0)
    Spontaneous sounds from 8 minimally speaking people at home, each labeled by a caregiver with what it meant.
    Tests the hard case: the label is a state of mind, so two different labels can be the same sound.

TORGO (Rudzicz, Namasivayam, Wolff 2012, doi:10.1007/s10579-011-9145-0, free for academic use)
    People with dysarthria from cerebral palsy or ALS, and matched controls, reading prompts. Repeats of one word
    stand in for a deliberate signal: the person means to make the same sound each time. The audio is not
    redistributed here; only aggregate results are.
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Homophones can't be told apart by sound, so keep just one of each pair.
HOMOPHONES = {"knew": "know", "new": "know", "no": "know", "write": "right", "meet": "meat"}
TORGO_MIN_REPEATS = 3  # 2 to enroll, at least 1 to test


@dataclass(frozen=True)
class Dataset:
    name: str
    root: Path
    paths: tuple[str, ...]  # relative to root
    labels: np.ndarray
    people: np.ndarray
    sessions: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)

    def __post_init__(self) -> None:
        n = len(self.paths)
        if not (len(self.labels) == len(self.people) == len(self.sessions) == n):
            raise ValueError(f"{self.name}: column lengths differ")
        if n == 0:
            raise ValueError(f"{self.name}: no clips found under {self.root}")


def _make(name: str, root: Path, rows: list[tuple[str, str, str, str]]) -> Dataset:
    paths, labels, people, sessions = zip(*rows, strict=True) if rows else ((), (), (), ())
    return Dataset(name, root, tuple(paths), np.array(labels), np.array(people), np.array(sessions))


def load_recanvo(root: Path) -> Dataset:
    with (root / "dataset_file_directory.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if (root / r["Filename"]).exists()]
    return _make("recanvo", root, [(r["Filename"], r["Label"], r["Participant"], r["Filename"][:11]) for r in rows])


def normalize_prompt(text: str) -> str | None:
    """The single word a prompt asks for, or None when it isn't one (sentences, instructions, unusable takes)."""
    text = re.sub(r"\[.*?\]", "", text).strip().lower().rstrip(".,!?")
    if not re.fullmatch(r"[a-z']+", text) or text == "xxx":
        return None
    return HOMOPHONES.get(text, text)


def is_dysarthric(person: str) -> bool:
    return "C" not in person  # F01, M03 have dysarthria; FC01, MC03 are controls


def load_torgo(root: Path) -> Dataset:
    rows = []
    for session in sorted(root.glob("*/Session*")):
        person = session.parent.name
        mic = next((m for m in ("wav_headMic", "wav_arrayMic") if any((session / m).glob("*.wav"))), None)
        if mic is None or not (session / "prompts").is_dir():
            continue
        for prompt in sorted((session / "prompts").glob("*.txt")):
            wav = session / mic / f"{prompt.stem}.wav"
            word = normalize_prompt(prompt.read_text(errors="ignore"))
            if word and wav.exists():
                rows.append((str(wav.relative_to(root)), word, person, f"{person}/{session.name}"))
    counts: dict[tuple[str, str], int] = {}
    for _, word, person, _ in rows:
        counts[person, word] = counts.get((person, word), 0) + 1
    return _make("torgo", root, [r for r in rows if counts[r[2], r[1]] >= TORGO_MIN_REPEATS])


LOADERS = {"recanvo": load_recanvo, "torgo": load_torgo}
