"""Metrics chosen for what a wrong answer costs a person, plus honest error bars.

With 7 or 8 people per benchmark, the person is the unit of analysis. Every interval resamples people, never clips:
thousands of clips from 8 people are still 8 people.
"""
import numpy as np

ECE_BINS = 10


def balanced_accuracy(scores: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall, so a common meaning can't hide a rare one. Chance is 1 / n_classes."""
    pred = scores.argmax(axis=1)
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(n_classes) if np.any(y == c)]))


def ranks(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1-based rank of the true class. This is how many yes/no questions Tink needs when it asks best guess first."""
    order = np.argsort(-scores, axis=1)
    return np.argmax(order == y[:, None], axis=1) + 1


def softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    z = scores / temp
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> float:
    """How far "I'm 80% sure" is from being right 80% of the time, averaged over equal-width bins."""
    bins = np.clip((confidence * ECE_BINS).astype(int), 0, ECE_BINS - 1)
    return float(sum(
        (bins == b).mean() * abs(correct[bins == b].mean() - confidence[bins == b].mean())
        for b in range(ECE_BINS) if np.any(bins == b)
    ))


def risk_coverage(confidence: np.ndarray, correct: np.ndarray, coverages=(1.0, 0.5, 0.25, 0.1)) -> dict[str, float]:
    """Accuracy if Tink answered only on the most confident fraction of sounds and asked on the rest."""
    order = np.argsort(-confidence)
    return {f"{c:g}": float(correct[order[: max(1, int(c * len(order)))]].mean()) for c in coverages}


def conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal cutoff: keep every class scoring above this and the true one is in the set with
    probability at least 1 - alpha, IF new sounds are exchangeable with the calibration ones."""
    n = len(calibration_scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(1 - calibration_scores, level, method="higher"))


def bootstrap_ci(per_person: np.ndarray, n: int = 10_000, seed: int = 0, level: float = 0.95) -> dict[str, float]:
    """Mean over people, with a percentile interval from resampling people."""
    per_person = np.asarray(per_person, dtype=float)
    rng = np.random.default_rng(seed)
    means = per_person[rng.integers(0, len(per_person), (n, len(per_person)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return {"mean": float(per_person.mean()), "low": float(np.percentile(means, tail)),
            "high": float(np.percentile(means, 100 - tail)), "n": len(per_person)}


def sign_flip_p(differences: np.ndarray) -> float:
    """Exact two-sided permutation test: could these paired differences come from a coin flip on each person's sign?

    A percentile bootstrap under-covers with 6 to 8 people, so a comparison has to pass this too. With n people the
    smallest possible p is 2 / 2^n, so 6 people can reach 0.031 at best.
    """
    d = np.asarray(differences, dtype=float)
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * len(d))).reshape(len(d), -1).T
    return float(np.mean(np.abs((signs * d).sum(axis=1)) >= abs(d.sum()) - 1e-12))


def beats(candidate: np.ndarray, baseline: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired comparison on the same people and the same episodes. Better means the CI is above zero AND p < alpha."""
    d = np.asarray(candidate) - np.asarray(baseline)
    ci = bootstrap_ci(d)
    p = sign_flip_p(d)
    return {**ci, "p": p, "better": bool(ci["low"] > 0 and p < alpha), "worse": bool(ci["high"] < 0 and p < alpha)}
