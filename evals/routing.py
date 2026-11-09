"""Local first, server when unsure: when should the on-device model hand a sound to a bigger model on a server?

    uv run --group eval python -m evals.routing --recanvo path/to/ReCANVo --torgo path/to/TORGO

Writes evals/results/routing.json. Run evals/onnx_gate.py on the pinned model first, as this reads its embeddings.

The device is the exact int8 file the browser downloads (the one pinned in web/engine/config.js). The server is
HuBERT in full precision, which is about 9 to 12 points better on spoken words and worse on non-verbal sounds. Both
see identical episodes. The device answers when its confidence is at least a threshold and the server answers the
rest. Confidence is the device's own: a softmax whose temperature is fitted on the enrolled takes alone (without the
app's augmented variants, like every other benchmark here), so no test label is used to decide where a sound goes.
"Sounds sent" is each person's share of their own sounds, averaged over people, and the budget uses the same mean.

Every threshold is also compared with routing the same number of sounds at random. That's the check that confidence
picks the right sounds to send, instead of any gain coming only from the server being better.

A fixed fallback budget is the realistic setting, as someone has to pay for the server. BUDGET was set before the
first run. Each person's threshold is picked from everyone else in the benchmark, never from that person, and the
report shows the fallback rate that person actually got, which is what a fleet would see.
"""
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evals import fewshot, metrics

OUT = Path(__file__).parent / "results" / "routing.json"
CONFIG = Path(__file__).parent.parent / "web" / "engine" / "config.js"
SERVER = "hubert"
ALWAYS = 1.01  # above any confidence, so every sound goes to the server
THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ALWAYS)
BUDGET = 0.3  # at most this share of sounds may go to the server
RANDOM_DRAWS = 10


@dataclass(frozen=True)
class Trial:
    """One episode's test sounds, as both models saw them."""

    device: np.ndarray  # predicted class per sound
    server: np.ndarray
    confidence: np.ndarray  # the device's
    truth: np.ndarray
    ways: int


def key(threshold: float) -> str:
    return f"{threshold:g}"


def route(device: np.ndarray, server: np.ndarray, confidence: np.ndarray,
          threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """The device keeps a sound if it's at least `threshold` sure. Returns the answers and which sounds were sent."""
    sent = confidence < threshold
    return np.where(sent, server, device), sent


def balanced(pred: np.ndarray, truth: np.ndarray, ways: int) -> float:
    """Mean per-class recall, the same as metrics.balanced_accuracy but from predictions."""
    return float(np.mean([np.mean(pred[truth == c] == c) for c in range(ways) if np.any(truth == c)]))


def person_trials(device_X: np.ndarray, server_X: np.ndarray, labels: np.ndarray, sessions: np.ndarray,
                  episode: fewshot.EpisodeFn, seed: int, trials: int) -> list[Trial]:
    """Same seed and episodes as every other benchmark, so this lines up with results.md."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        ep = episode(labels, sessions, rng)
        enroll, test = np.concatenate(ep.enroll), np.concatenate(ep.test)
        ye = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.enroll)])
        yt = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.test)])
        S = fewshot.centroid(device_X[enroll], ye, device_X[test])
        loo, loo_y = fewshot._leave_one_take_out(device_X[enroll], ye)
        conf = metrics.softmax(S, fewshot._fit_temp(loo, loo_y)).max(axis=1)
        server = fewshot.centroid(server_X[enroll], ye, server_X[test]).argmax(axis=1)
        out.append(Trial(S.argmax(axis=1), server, conf, yt, len(ep.enroll)))
    return out


def score_person(trials: list[Trial], thresholds=THRESHOLDS, seed: int = 0) -> dict:
    """Per threshold: accuracy, share of sounds sent, and accuracy when that many are sent at random instead."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in thresholds:
        accs, rand, sent_n, total = [], [], 0, 0
        for tr in trials:
            pred, sent = route(tr.device, tr.server, tr.confidence, t)
            accs.append(balanced(pred, tr.truth, tr.ways))
            k = int(sent.sum())
            sent_n, total = sent_n + k, total + len(sent)
            draws = []
            for _ in range(RANDOM_DRAWS):
                mask = np.zeros(len(sent), bool)
                mask[rng.choice(len(sent), k, replace=False)] = True
                draws.append(balanced(np.where(mask, tr.server, tr.device), tr.truth, tr.ways))
            rand.append(np.mean(draws))
        out[key(t)] = {"accuracy": float(np.mean(accs)), "fallback": sent_n / total, "random": float(np.mean(rand))}
    return out


def pick_threshold(fallback: dict, held_out: str, budget: float = BUDGET) -> str:
    """The highest threshold whose mean fallback rate over everyone except `held_out` stays within budget.

    `fallback` maps person -> threshold key -> fallback rate. Returns "0" (never send) if nothing fits.
    """
    others = [p for p in fallback if p != held_out]
    fits = [k for k in next(iter(fallback.values()))
            if np.mean([fallback[p][k] for p in others]) <= budget]
    return max(fits, key=float, default="0")


def summarize(per_person: dict) -> dict:
    """Mean over people per threshold, plus the leave-one-person-out budget row."""
    people = sorted(per_person)
    device = np.array([per_person[p][key(0.0)]["accuracy"] for p in people])
    out: dict = {"people": len(people), "thresholds": {}}
    for k in per_person[people[0]]:
        acc = np.array([per_person[p][k]["accuracy"] for p in people])
        rand = np.array([per_person[p][k]["random"] for p in people])
        out["thresholds"][k] = {
            "accuracy": metrics.bootstrap_ci(acc),
            "fallback": float(np.mean([per_person[p][k]["fallback"] for p in people])),
            "random": float(rand.mean()),
            "vs_device": metrics.beats(acc, device),
            "vs_random": metrics.beats(acc, rand),
        }
    rates = {p: {k: r["fallback"] for k, r in per_person[p].items()} for p in people}
    picked = {p: pick_threshold(rates, p) for p in people}
    acc = np.array([per_person[p][picked[p]]["accuracy"] for p in people])
    rand = np.array([per_person[p][picked[p]]["random"] for p in people])
    server = float(np.mean([per_person[p][key(ALWAYS)]["accuracy"] for p in people]))
    gap = server - device.mean()
    out["budget"] = {
        "budget": BUDGET,
        "threshold_per_person": picked,
        "fallback_per_person": {p: round(per_person[p][picked[p]]["fallback"], 4) for p in people},
        "fallback": float(np.mean([per_person[p][picked[p]]["fallback"] for p in people])),
        "accuracy": metrics.bootstrap_ci(acc),
        "device": float(device.mean()),
        "server": server,
        "share_of_server_gain": float((acc.mean() - device.mean()) / gap) if gap > 0 else None,
        "random": float(rand.mean()),
        "vs_device": metrics.beats(acc, device),
        "vs_random": metrics.beats(acc, rand),
    }
    return out


def pinned_model() -> str:
    """The cache name evals/onnx_gate.py gives the model the browser pins, so this scores that exact file."""
    config = CONFIG.read_text()
    sha = re.search(r'sha256: "([0-9a-f]{64})"', config)
    tag = re.search(r'tag: "([^"]+)"', config)
    if not sha or not tag:
        raise ValueError(f"no pinned model in {CONFIG}")
    return f"{tag.group(1)}-{sha.group(1)[:12]}"


def main() -> None:
    from evals import run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recanvo", type=Path, required=True)
    ap.add_argument("--torgo", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    device_model = pinned_model()
    report: dict = {"device": device_model, "server": f"`{run.MODELS[SERVER]}` layer {run.APP_LAYER} in full precision",
                    "budget": BUDGET, "benchmarks": {}}
    for b in run._build(args):
        cache = b.aug_path.with_name(f"{device_model}.npy")  # same folder as every other cached embedding
        if not cache.exists():
            raise SystemExit(f"missing {cache}: run evals/onnx_gate.py on the pinned model first")
        device_all = np.load(cache, mmap_mode="r")
        per = {}
        for p, (idx, labels, sessions) in b.people.items():
            trials = person_trials(np.asarray(device_all[idx], np.float32), run._matrix(b, SERVER, run.APP_LAYER, p),
                                   labels, sessions, b.episode, fewshot.person_seed(p), b.trials)
            per[p] = score_person(trials, seed=fewshot.person_seed(p))
        report["benchmarks"][b.name] = summarize(per)
        r = report["benchmarks"][b.name]["budget"]
        print(f"{b.name}: device {r['device']:.3f} server {r['server']:.3f} "
              f"budget {r['accuracy']['mean']:.3f} at {r['fallback']:.2f} sent, random {r['random']:.3f}", flush=True)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
