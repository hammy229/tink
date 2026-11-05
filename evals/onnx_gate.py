"""Does the model that actually ships score the same as the one that was evaluated?

    uv run --group eval --group export python -m evals.onnx_gate models/voc2vec-l12.int8.onnx --layer 12 \\
        --recanvo path/to/ReCANVo --torgo path/to/TORGO

Cosine agreement (scripts/export_onnx.py) can look fine while accuracy moves, so this re-embeds the real benchmarks
with the exported file and compares accuracy per person against PyTorch on identical episodes. It exits with an error
if the build loses more than MAX_LOSS on any benchmark. tests/test_evals.py then fails unless the hash in
web/engine/config.js has a passing report here, so a build with no passing gate can't be pinned.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from evals import fewshot, metrics
from evals.datasets import is_dysarthric, load_recanvo, load_torgo
from evals.embed import cache_path
from tink import audio
from tink.embed import preprocess

MAX_LOSS = 0.01  # accuracy points, mean over people
OUT = Path(__file__).parent / "results"


def embed_all(ds, path: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.zeros((len(ds), 768), np.float16)
    t0 = time.perf_counter()
    for i, name in enumerate(ds.paths):
        wav = audio.load(ds.root / name)[: 30 * audio.SR]
        out[i] = session.run(None, {"input_values": preprocess(wav)[None]})[0][0]
        if i % 1000 == 0:
            print(f"{ds.name} {i}/{len(ds)} {time.perf_counter() - t0:.0f}s", flush=True)
    np.save(cache, out)
    return out


def compare(name: str, people: dict, reference: np.ndarray, exported: np.ndarray, layer: int, episode,
            trials: int) -> dict:
    ref, exp = {}, {}
    for p, (idx, labels, sessions) in people.items():
        seed = fewshot.person_seed(p)
        ref[p] = fewshot.run_person(np.asarray(reference[idx, layer], np.float32), labels, sessions, fewshot.centroid,
                                    episode, seed, trials)["accuracy"]
        exp[p] = fewshot.run_person(np.asarray(exported[idx], np.float32), labels, sessions, fewshot.centroid,
                                    episode, seed, trials)["accuracy"]
    a, b = np.array(list(exp.values())), np.array(list(ref.values()))
    return {"benchmark": name, "exported": float(a.mean()), "pytorch": float(b.mean()), "diff": metrics.beats(a, b),
            "passes": bool(b.mean() - a.mean() <= MAX_LOSS)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--recanvo", type=Path, required=True)
    ap.add_argument("--torgo", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    recanvo, torgo = load_recanvo(args.recanvo), load_torgo(args.torgo)
    sha = _sha256(args.model)
    stem = f"{args.model.stem}-{sha[:12]}"  # keyed by content, so a new export can't pass on old embeddings
    exported_r = embed_all(recanvo, args.model, cache_path(recanvo, stem))
    exported_t = embed_all(torgo, args.model, cache_path(torgo, stem))
    ref_r = np.load(cache_path(recanvo, "voc2vec"), mmap_mode="r")
    ref_t = np.load(cache_path(torgo, "voc2vec"), mmap_mode="r")

    rec = {}
    for p in sorted(set(recanvo.people)):
        idx = np.flatnonzero(recanvo.people == p)
        keep = fewshot.usable_for_sessions(recanvo.labels[idx], recanvo.sessions[idx])
        if len(set(recanvo.labels[idx][keep])) >= 2:
            rec[p] = (idx[keep], recanvo.labels[idx][keep], recanvo.sessions[idx][keep])
    dys = {}
    for p in sorted(set(torgo.people)):
        idx = np.flatnonzero(torgo.people == p)
        if is_dysarthric(p) and len(set(torgo.labels[idx])) >= fewshot.WORD_WAYS:
            dys[p] = (idx, torgo.labels[idx], torgo.sessions[idx])

    results = [
        compare("recanvo", rec, ref_r, exported_r, args.layer, fewshot.across_sessions, args.trials),
        compare("dysarthric", dys, ref_t, exported_t, args.layer, fewshot.word_episode, args.trials * 4),
    ]
    report = {"model": args.model.name, "sha256": sha, "bytes": args.model.stat().st_size,
              "layer": args.layer, "max_loss": MAX_LOSS,
              "results": results, "passes": all(r["passes"] for r in results)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"gate-{args.model.stem}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passes"]:
        sys.exit("GATE FAILED: the exported model loses accuracy")


if __name__ == "__main__":
    main()
