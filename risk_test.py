"""Risk test: can frozen voc2vec embeddings tell one person's sounds apart?

Record takes through the app (Teach a new sound) or drop WAVs into data/takes/<sound>/*.wav, then:
    uv run python risk_test.py [data/takes]

For every layer, with and without centering, runs leave-one-out nearest-centroid and prints accuracy.
Then prints the score spread at the best setting so you can pick TINK_LAYER and TINK_THRESHOLD.
"""
import sys
from pathlib import Path

import numpy as np

from tink import audio
from tink.embed import MODEL_ID, layers


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def loo(X, y, center):
    """Leave-one-out nearest centroid. Returns (accuracy, correct-class scores, best wrong-class scores)."""
    hits, right, wrong = 0, [], []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        Xk, yk = X[keep], y[keep]
        mu = Xk.mean(axis=0) if center else 0
        classes = sorted(set(yk))
        cents = unit(np.stack([Xk[yk == c].mean(axis=0) - mu for c in classes]))
        s = cents @ unit(X[i] - mu)
        hits += classes[int(s.argmax())] == y[i]
        right.append(s[classes.index(y[i])])
        wrong.append(max(v for c, v in zip(classes, s) if c != y[i]))
    return hits / len(y), np.array(right), np.array(wrong)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/takes")
    files = sorted(p for p in root.glob("*/*") if p.suffix.lower() in {".wav", ".flac"})
    y = np.array([p.parent.name for p in files])
    counts = {c: int((y == c).sum()) for c in sorted(set(y))}
    if len(counts) < 2 or min(counts.values()) < 2:
        sys.exit(f"Need at least 2 sounds with 2+ takes each in {root}. Found: {counts or 'nothing'}")

    print(f"{MODEL_ID}\n{len(files)} clips: {counts}\n")
    H = np.stack([layers(audio.load(p)) for p in files])  # (clips, layers, dim)

    print("layer   raw    centered")
    best = (0, None, None)
    for L in range(H.shape[1]):
        row = []
        for center in (False, True):
            acc, *_ = loo(H[:, L], y, center)
            row.append(acc)
            if acc > best[0]:
                best = (acc, L, center)
        print(f"{L:>5}   {row[0]:>4.0%}   {row[1]:>4.0%}")

    acc, L, center = best
    _, right, wrong = loo(H[:, L], y, True)  # the app always centers
    print(f"\nBest: layer {L}, {'centered' if center else 'raw'}, accuracy {acc:.0%}")
    print(f"  centered scores at layer {L}:")
    print(f"  correct class   min {right.min():.3f}  median {np.median(right):.3f}")
    print(f"  best wrong      max {wrong.max():.3f}  median {np.median(wrong):.3f}")
    print(f"  suggested threshold  {(np.median(right) + np.median(wrong)) / 2:.2f}")
    print(f"\nUse it:  TINK_LAYER={L} TINK_THRESHOLD=<threshold> uv run uvicorn tink.app:app --reload")
    print("Go if accuracy >= 85%. Below that: try another layer or model, or make the sounds more distinct.")


if __name__ == "__main__":
    main()
