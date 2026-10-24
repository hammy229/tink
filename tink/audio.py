"""Load any WAV/FLAC into 16 kHz mono float32 and trim the silence around the sound."""
import io
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000
MIN_SECONDS = 0.4


def load(src: bytes | str | Path) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(src) if isinstance(src, bytes) else src, dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return trim(wav)


def trim(wav: np.ndarray, frame: int = 320, rel: float = 0.1, pad: float = 0.1) -> np.ndarray:
    """Keep the span between the first and last 20 ms frame louder than `rel` x the loudest frame."""
    n = len(wav) // frame
    if n:
        rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        loud = np.flatnonzero(rms > rms.max() * rel)
        if loud.size:
            p = int(pad * SR)
            wav = wav[max(0, loud[0] * frame - p) : (loud[-1] + 1) * frame + p]
    short = int(MIN_SECONDS * SR) - len(wav)
    return np.pad(wav, (0, short)) if short > 0 else wav


def is_silent(wav: np.ndarray, floor: float = 0.005) -> bool:
    """True when no 20 ms frame is louder than `floor` RMS: nothing to learn from."""
    n = len(wav) // 320
    return n == 0 or float(np.sqrt((wav[: n * 320].reshape(n, 320) ** 2).mean(axis=1)).max()) < floor


def to_wav_bytes(wav: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()
