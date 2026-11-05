"""Synthetic audio shared by the tests, so no recordings are needed."""
import io

import numpy as np
import soundfile as sf


def tone(freq: float, seconds: float = 0.8, sr: int = 16_000, noise: float = 0.003, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    wav = 0.5 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t)) + rng.normal(0, noise, len(t))
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def hiss(seconds: float = 0.8, sr: int = 16_000, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    wav = rng.normal(0, 0.3, int(sr * seconds)) * np.hanning(int(sr * seconds))
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV")
    return buf.getvalue()
