"""Turn every clip of a benchmark into cached features, once.

    uv run --group eval python -m evals.embed recanvo path/to/ReCANVo
    uv run --group eval python -m evals.embed torgo   path/to/TORGO

Each model's output is (clips, layers, dim) in float16. A log-mel baseline is included so the deep models have to
beat something simple. A clip that can't be read is recorded as zeros and reported, never silently dropped.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window

from evals.datasets import LOADERS, Dataset
from tink import audio

MODELS = {
    "voc2vec": "alkiskoudounas/voc2vec-hubert-ls-pt",
    "hubert": "facebook/hubert-base-ls960",
    "wav2vec2": "facebook/wav2vec2-base",
}
LOGMEL = "logmel"
N_MELS, N_FFT, HOP = 40, 400, 160


def _mel_filters() -> np.ndarray:
    to_mel, to_hz = (lambda f: 2595 * np.log10(1 + f / 700)), (lambda m: 700 * (10 ** (m / 2595) - 1))
    edges = to_hz(np.linspace(to_mel(0), to_mel(audio.SR / 2), N_MELS + 2))
    freqs = np.linspace(0, audio.SR / 2, N_FFT // 2 + 1)
    up = (freqs[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    down = (edges[2:, None] - freqs[None]) / (edges[2:, None] - edges[1:-1, None])
    return np.clip(np.minimum(up, down), 0, None)


FILTERS = _mel_filters()
WINDOW = get_window("hann", N_FFT)


def logmel_stats(wav: np.ndarray) -> np.ndarray:
    """Mean and standard deviation over time of MFCCs and their deltas: the classic hand-built speech feature."""
    wav = np.pad(wav, (0, max(0, N_FFT - len(wav))))
    frames = np.lib.stride_tricks.sliding_window_view(wav, N_FFT)[::HOP] * WINDOW
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mfcc = dct(np.log(power @ FILTERS.T + 1e-8), axis=1, norm="ortho")[:, :20]
    delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])
    stats = np.concatenate([mfcc.mean(0), mfcc.std(0), delta.mean(0), delta.std(0)])
    return (stats / (np.linalg.norm(stats) + 1e-9)).astype(np.float32)[None]


def cache_path(ds: Dataset, model: str) -> Path:
    return ds.root / "embeddings" / f"{model}.npy"


def embed_dataset(ds: Dataset, model: str) -> None:
    out = cache_path(ds, model)
    if out.exists():
        print(f"{ds.name}/{model}: cached")
        return
    if model == LOGMEL:
        extract = logmel_stats
    else:
        from tink.embed import layers

        extract = lambda wav: layers(wav, MODELS[model])
    first = extract(np.zeros(audio.SR, np.float32))
    H = np.zeros((len(ds), *first.shape), np.float16)
    failed = []
    for i, name in enumerate(ds.paths):
        try:
            H[i] = extract(audio.load(ds.root / name)[: 30 * audio.SR])
        except Exception as e:
            failed.append((name, repr(e)))
        if i % 500 == 0:
            print(f"{ds.name}/{model} {i}/{len(ds)}", flush=True)
    out.parent.mkdir(exist_ok=True)
    np.save(out, H)
    print(f"{ds.name}/{model}: done, {len(failed)} unreadable", flush=True)
    for name, err in failed[:20]:
        print("  ", name, err)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in LOADERS:
        sys.exit(__doc__)
    dataset = LOADERS[sys.argv[1]](Path(sys.argv[2]))
    print(f"{dataset.name}: {len(dataset)} clips, {len(set(dataset.people))} people, {len(set(dataset.labels))} labels")
    for m in (LOGMEL, *MODELS):
        embed_dataset(dataset, m)
