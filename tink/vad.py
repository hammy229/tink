"""Hands-free listening: find where each sound starts and ends in a continuous stream.

Someone with advanced ALS can't press a Listen button, so the stream has to segment itself. A fixed loudness threshold
fails the moment a fan turns on, so this one tracks the room's noise floor and fires on energy above it.

The rules are mirrored in web/engine/core.js and tested against the same fixture (tests/fixtures/parity.json).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

FRAME = 320  # 20 ms at 16 kHz


@dataclass(frozen=True)
class VadConfig:
    ratio: float = 3.0  # a frame is sound when it is this many times the noise floor's RMS (about 10 dB)
    floor_min: float = 0.002  # never let a silent room make the threshold infinitely sensitive
    floor_rise: float = 0.02  # between sounds the floor rises gently toward the room
    floor_rise_loud: float = 0.002  # during loud frames, far slower: a 4 s hum can't teach it, but a steady fan will
    floor_fall: float = 0.2  # and drops quickly when the room gets quieter
    start_frames: int = 3  # 60 ms above threshold to start, so a door tick doesn't
    end_frames: int = 25  # 500 ms below threshold to end, so two clicks stay one sound
    pre_roll: int = 10  # 200 ms kept from before the start, to keep the onset
    max_frames: int = 200  # 4 s: cut off a sound that never ends
    min_frames: int = 5  # 100 ms: shorter segments are dropped


class SegmentDetector:
    """Feed samples in any chunk size; get back finished sounds as they end."""

    def __init__(self, cfg: VadConfig | None = None):
        self.cfg = cfg or VadConfig()
        self.floor: float | None = None
        self._buffer = np.zeros(0, np.float32)
        self._before: deque[np.ndarray] = deque(maxlen=self.cfg.pre_roll + self.cfg.start_frames)
        self._current: list[np.ndarray] | None = None
        self._above = 0
        self._quiet = 0
        self._loud = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        n = len(self._buffer) // FRAME
        out = []
        for i in range(n):
            done = self._frame(self._buffer[i * FRAME : (i + 1) * FRAME])
            if done is not None:
                out.append(done)
        self._buffer = self._buffer[n * FRAME :]
        return out

    def flush(self) -> list[np.ndarray]:
        """The stream ended: hand over any sound still in progress."""
        done = self._finish() if self._current is not None else None
        return [done] if done is not None else []

    def _threshold(self) -> float:
        return max(self.cfg.floor_min, self.floor or 0.0) * self.cfg.ratio

    def _track_floor(self, rms: float, loud: bool) -> None:
        if self.floor is None:
            self.floor = rms
        elif rms < self.floor:
            self.floor += (rms - self.floor) * self.cfg.floor_fall
        else:
            self.floor += (rms - self.floor) * (self.cfg.floor_rise_loud if loud else self.cfg.floor_rise)

    def _frame(self, frame: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(frame**2)))
        loud = rms > self._threshold()
        self._track_floor(rms, loud)
        if self._current is None:
            self._above = self._above + 1 if loud else 0
            self._before.append(frame)
            if self._above >= self.cfg.start_frames:
                self._current = list(self._before)
                self._before.clear()
                self._loud = self._above
                self._quiet = self._above = 0
            return None
        self._current.append(frame)
        if loud:
            self._quiet = 0
            self._loud += 1
        else:
            self._quiet += 1
        if self._quiet >= self.cfg.end_frames or len(self._current) >= self.cfg.max_frames:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray | None:
        frames = self._current or []
        self._current = None
        self._quiet = 0
        return np.concatenate(frames) if self._loud >= self.cfg.min_frames else None
