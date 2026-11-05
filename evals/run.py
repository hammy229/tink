"""Run the benchmarks and write evals/results/report.json. Nothing in the report is typed by hand.

    uv run --group eval python -m evals.embed recanvo ~/Datasets/recanvo/extracted    # once
    uv run --group eval python -m evals.embed torgo ~/Datasets/torgo                  # once
    uv run --group eval python -m evals.run --recanvo ~/Datasets/recanvo/extracted --torgo ~/Datasets/torgo

Benchmarks
  recanvo     spontaneous sounds; 5 takes per meaning; test only on other sessions; each meaning can be taught on a
              different day
  recanvo_one_sitting
              the same, but taught in one session, covering only the meanings that session has
  dysarthric  TORGO speakers with dysarthria; 5 words at a time, 2 takes per word, any session
  controls    TORGO matched controls; same task, shows what dysarthria costs
  dysarthric_other_session, controls_other_session
              the same, but tested only on takes from a different session: teach once, use on another day

Promotion gate for the trained head, set before it was run: ship it only if it beats the app's method on BOTH recanvo
and dysarthric with a paired 95% CI above zero.
"""
import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from evals import fewshot, head, metrics
from evals.datasets import is_dysarthric, load_recanvo, load_torgo
from evals.embed import LOGMEL, MODELS, cache_path

OUT = Path(__file__).parent / "results" / "report.json"
PRIMARY = "voc2vec"
APP_LAYER = 12
TRAINING_POOL = ("recanvo", "dysarthric", "controls")  # the *_other_session benches reuse the same people


@dataclass
class Bench:
    """One benchmark: each person's clips and how to sample their episodes."""

    name: str
    people: dict  # person -> (indices, labels, sessions)
    episode: fewshot.EpisodeFn
    trials: int
    feats: dict  # model -> memmapped (clips, layers, dim)
    aug_path: Path
    paths: np.ndarray  # names sort by recording time within a person


def _build(args) -> list[Bench]:
    recanvo = load_recanvo(args.recanvo)
    rec = {}
    for p in sorted(set(recanvo.people)):
        idx = np.flatnonzero(recanvo.people == p)
        keep = fewshot.usable_for_sessions(recanvo.labels[idx], recanvo.sessions[idx])
        if len(set(recanvo.labels[idx][keep])) >= 2:
            rec[p] = (idx[keep], recanvo.labels[idx][keep], recanvo.sessions[idx][keep])
    feats_r = {m: np.load(cache_path(recanvo, m), mmap_mode="r") for m in (LOGMEL, *MODELS)}

    torgo = load_torgo(args.torgo)
    groups: dict[str, dict] = {k: {} for k in ("dysarthric", "controls", "dysarthric_other_session",
                                               "controls_other_session")}
    for p in sorted(set(torgo.people)):
        idx = np.flatnonzero(torgo.people == p)
        if len(set(torgo.labels[idx])) >= fewshot.WORD_WAYS:
            group = "dysarthric" if is_dysarthric(p) else "controls"
            groups[group][p] = (idx, torgo.labels[idx], torgo.sessions[idx])
            if fewshot.enrollment_sessions(torgo.labels[idx], torgo.sessions[idx]):
                groups[f"{group}_other_session"][p] = (idx, torgo.labels[idx], torgo.sessions[idx])
    feats_t = {m: np.load(cache_path(torgo, m), mmap_mode="r") for m in (LOGMEL, *MODELS)}
    aug_r, aug_t = cache_path(recanvo, "voc2vec-aug"), cache_path(torgo, "voc2vec-aug")
    paths_r, paths_t = np.array(recanvo.paths), np.array(torgo.paths)
    one_sitting = {p: v for p, v in rec.items() if fewshot.one_sitting_meanings(v[1], v[2])}
    return [
        Bench("recanvo", rec, fewshot.across_sessions, args.trials, feats_r, aug_r, paths_r),
        Bench("recanvo_one_sitting", one_sitting, fewshot.across_sessions_one_sitting, args.trials, feats_r, aug_r,
              paths_r),
        Bench("dysarthric", groups["dysarthric"], fewshot.word_episode, args.trials * 4, feats_t, aug_t, paths_t),
        Bench("controls", groups["controls"], fewshot.word_episode, args.trials * 4, feats_t, aug_t, paths_t),
        Bench("dysarthric_other_session", groups["dysarthric_other_session"], fewshot.word_episode_across_sessions,
              args.trials * 4, feats_t, aug_t, paths_t),
        Bench("controls_other_session", groups["controls_other_session"], fewshot.word_episode_across_sessions,
              args.trials * 4, feats_t, aug_t, paths_t),
    ]


def _matrix(bench: Bench, model: str, layer: int, person: str) -> np.ndarray:
    idx = bench.people[person][0]
    return np.asarray(bench.feats[model][idx, layer], dtype=np.float32)


def _score(bench: Bench, model: str, layer: int, method: fewshot.Method, transform=None) -> dict:
    out = {}
    for p, (_, labels, sessions) in bench.people.items():
        X = _matrix(bench, model, layer, p)
        if transform is not None:
            X = transform(p, X)
        out[p] = fewshot.run_person(X, labels, sessions, method, bench.episode, fewshot.person_seed(p), bench.trials)
    return out


def _summary(per_person: dict) -> dict:
    return {
        **metrics.bootstrap_ci(np.array([r["accuracy"] for r in per_person.values()])),
        "chance": float(np.mean([r["chance"] for r in per_person.values()])),
        "mean_questions": float(np.mean([r["mean_questions"] for r in per_person.values()])),
        "within_2": float(np.mean([r["within_2"] for r in per_person.values()])),
        "within_3": float(np.mean([r["within_3"] for r in per_person.values()])),
        "per_person": {p: round(r["accuracy"], 4) for p, r in per_person.items()},
    }


def _acc(per_person: dict) -> np.ndarray:
    return np.array([r["accuracy"] for _, r in sorted(per_person.items())])


def compare_models_and_methods(benches: list[Bench]) -> dict:
    report = {}
    for b in benches:
        res: dict = {"people": len(b.people), "trials": b.trials}
        base = _score(b, PRIMARY, APP_LAYER, fewshot.centroid)
        res["baseline"] = _summary(base)
        res["models"] = {}
        for model in (LOGMEL, *MODELS):
            layer = 0 if model == LOGMEL else APP_LAYER
            r = _score(b, model, layer, fewshot.centroid)
            res["models"][model] = {**_summary(r), "vs_baseline": metrics.beats(_acc(r), _acc(base))}
        res["layers"] = {}
        for layer in range(b.feats[PRIMARY].shape[1]):
            r = _score(b, PRIMARY, layer, fewshot.centroid)
            res["layers"][layer] = {**_summary(r), "vs_baseline": metrics.beats(_acc(r), _acc(base))}
        res["methods"] = {}
        for name, method in fewshot.METHODS.items():
            r = _score(b, PRIMARY, APP_LAYER, method)
            res["methods"][name] = {**_summary(r), "vs_baseline": metrics.beats(_acc(r), _acc(base))}
        report[b.name] = res
        print(f"{b.name}: baseline {res['baseline']['mean']:.3f} chance {res['baseline']['chance']:.3f}", flush=True)
    return report


def uncertainty(benches: list[Bench]) -> dict:
    out = {}
    for b in benches:
        per = {p: fewshot.run_uncertainty(_matrix(b, PRIMARY, APP_LAYER, p), labels, sessions, b.episode,
                                          fewshot.person_seed(p), b.trials)
               for p, (_, labels, sessions) in b.people.items()}
        keys = [k for k, v in next(iter(per.values())).items() if not isinstance(v, dict)]
        out[b.name] = {k: float(np.mean([r[k] for r in per.values()])) for k in keys}
        out[b.name]["risk_coverage"] = {
            c: float(np.mean([r["risk_coverage"][c] for r in per.values()]))
            for c in next(iter(per.values()))["risk_coverage"]
        }
    return out


def augmentation(benches: list[Bench]) -> dict:
    """The app enrolls each take plus 4 variants. Is that worth 5 times the model calls?"""
    out = {}
    for b in benches:
        aug = np.load(b.aug_path, mmap_mode="r")
        off, on = {}, {}
        for p, (idx, labels, sessions) in b.people.items():
            X = _matrix(b, PRIMARY, APP_LAYER, p)
            seed = fewshot.person_seed(p)
            off[p] = fewshot.run_person(X, labels, sessions, fewshot.centroid, b.episode, seed, b.trials)
            on[p] = fewshot.run_person(X, labels, sessions, fewshot.centroid, b.episode, seed, b.trials,
                                       augmented=np.asarray(aug[idx], np.float32))
        out[b.name] = {"off": _summary(off), "on": _summary(on), "vs_off": metrics.beats(_acc(on), _acc(off))}
    return out


def adaptive(benches: list[Bench]) -> dict:
    """Exploratory: same bar as the head, but the design was fixed after reading these same benchmarks. Learning from
    "Right" stays only if later sounds benefit on spontaneous sounds and on deliberate sounds. Deliberate pools every
    TORGO speaker across sessions, as dysarthric speakers alone are too few for the sign-flip test to ever pass."""
    out: dict = {}
    pooled: list[tuple[float, float]] = []
    for b in benches:
        if b.name not in ("recanvo", "dysarthric_other_session", "controls_other_session"):
            continue
        ran = {p: fewshot.run_adaptive(_matrix(b, PRIMARY, APP_LAYER, p), labels, sessions, b.episode,
                                       fewshot.person_seed(p), b.trials,
                                       recorded_at=np.argsort(np.argsort(b.paths[idx])))
               for p, (idx, labels, sessions) in b.people.items()}
        per = {p: r for p, r in ran.items() if r is not None}
        static = np.array([r["static"] for r in per.values()])
        grown = np.array([r["adaptive"] for r in per.values()])
        out[b.name] = {"people": len(per), "static": float(static.mean()), "adaptive": float(grown.mean()),
                       "vs_static": metrics.beats(grown, static),
                       "wrong_clips_per_episode": float(np.mean([r["wrong_clips_per_episode"] for r in per.values()])),
                       "confirmed_clips_per_episode": float(np.mean([r["confirmed_clips_per_episode"]
                                                                     for r in per.values()]))}
        if b.name != "recanvo":
            pooled += list(zip(grown.tolist(), static.tolist(), strict=True))
    both = np.array(pooled)
    out["all_torgo_other_session"] = {"people": len(both), "static": float(both[:, 1].mean()),
                                      "adaptive": float(both[:, 0].mean()),
                                      "vs_static": metrics.beats(both[:, 0], both[:, 1])}
    out["passes_gate"] = out["recanvo"]["vs_static"]["better"] and out["all_torgo_other_session"]["vs_static"]["better"]
    return out


def _tasks(b: Bench, layer: int, exclude: set) -> dict:
    return {p: head.Task(torch.from_numpy(_matrix(b, PRIMARY, layer, p).copy()), labels, sessions, b.episode)
            for p, (_, labels, sessions) in b.people.items() if p not in exclude}


def _project_with(heads: dict):
    return lambda person, X: head.apply(heads[person], X)


def leave_one_person_out_head(benches: list[Bench], layer: int, seeds: tuple[int, ...]) -> dict:
    """Train on everyone but the tested person, stop early on one more held-out person, across all benchmarks."""
    results: dict = {}
    by_name = {b.name: b for b in benches}
    for target in ("recanvo", "dysarthric"):
        b = by_name[target]
        base = _score(b, PRIMARY, layer, fewshot.centroid)
        per_seed = []
        steps = []
        for seed in seeds:
            cfg = head.HeadConfig(seed=seed)
            heads = {}
            for held in b.people:
                pool = {f"{o.name}/{p}": t for o in benches if o.name in TRAINING_POOL
                        for p, t in _tasks(o, layer, {held}).items()}
                keys = sorted(pool)
                val_key = np.random.default_rng(seed + fewshot.person_seed(held)).choice(
                    [k for k in keys if k.startswith(target)])
                train = [pool[k] for k in keys if k != val_key]
                heads[held], info = head.train_head(train, [pool[val_key]], cfg)
                steps.append(info["steps"])
            r = _score(b, PRIMARY, layer, fewshot.centroid, transform=_project_with(heads))
            per_seed.append(r)
        mean_acc = {p: {"accuracy": float(np.mean([s[p]["accuracy"] for s in per_seed])),
                        "chance": per_seed[0][p]["chance"],
                        "mean_questions": float(np.mean([s[p]["mean_questions"] for s in per_seed])),
                        "within_2": float(np.mean([s[p]["within_2"] for s in per_seed])),
                        "within_3": float(np.mean([s[p]["within_3"] for s in per_seed]))} for p in b.people}
        results[target] = {**_summary(mean_acc), "baseline": _summary(base), "vs_baseline": metrics.beats(
            _acc(mean_acc), _acc(base)), "mean_steps": float(np.mean(steps))}
        print(f"head on {target}: {results[target]['mean']:.3f} vs {results[target]['baseline']['mean']:.3f}",
              flush=True)
    results["config"] = asdict(head.HeadConfig())
    results["seeds"] = list(seeds)
    results["layer"] = layer
    results["passes_gate"] = all(results[t]["vs_baseline"]["better"] for t in ("recanvo", "dysarthric"))
    return results


def leakage_check(benches: list[Bench]) -> dict:
    """Shuffle the labels: a score above chance means the scoring itself is broken. This can't catch session or mic
    leakage, as those survive a shuffle; the session-disjoint benchmarks are what address that."""
    out = {}
    for b in benches:
        rows = []
        for p, (_, labels, sessions) in b.people.items():
            shuffled = np.random.default_rng(0).permutation(labels)
            usable = np.ones(len(labels), bool)
            if b.name == "recanvo":
                usable = fewshot.usable_for_sessions(shuffled, sessions)
                if len(set(shuffled[usable])) < 2:
                    continue
            X = _matrix(b, PRIMARY, APP_LAYER, p)[usable]
            try:
                rows.append(fewshot.run_person(X, shuffled[usable], sessions[usable], fewshot.centroid, b.episode,
                                               fewshot.person_seed(p), b.trials))
            except ValueError:  # shuffling can leave this person with no episode that meets the benchmark's rules
                continue
        if not rows:
            continue
        out[b.name] = {"accuracy": float(np.mean([r["accuracy"] for r in rows])),
                       "chance": float(np.mean([r["chance"] for r in rows]))}
        if out[b.name]["accuracy"] > out[b.name]["chance"] + 0.05:
            raise RuntimeError(f"{b.name}: shuffled labels scored above chance, so the evaluation is broken")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recanvo", type=Path, required=True)
    ap.add_argument("--torgo", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--head-layer", type=int, default=APP_LAYER)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    benches = _build(args)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {
        "git_sha": sha,
        "benchmarks": compare_models_and_methods(benches),
        "leakage_check": leakage_check(benches),
        "augmentation": augmentation(benches),
        "uncertainty": uncertainty(benches),
        "adaptive": adaptive(benches),
        "head": leave_one_person_out_head(benches, args.head_layer, tuple(range(args.seeds))),
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
