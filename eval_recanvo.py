"""Accuracy on real nonspeaking people: ReCANVo (Narain and Johnson 2021, doi:10.5281/zenodo.5786860, CC BY 3.0).

7,077 sounds from 8 minimally speaking people, each labeled by a caregiver with what it meant
(frustrated, delighted, request, yes, ...). This runs the app's setup on them: per person,
5 examples per meaning, centered nearest centroid, then guess the meaning of sounds it hasn't heard.

    uv run python eval_recanvo.py embed  path/to/ReCANVo   # once per model, caches every layer
    uv run python eval_recanvo.py score  path/to/ReCANVo

Two splits:
  other sessions  enroll 5 clips from one recording session, test only on clips from other sessions
                  (sounds from one sitting are alike, so this is the honest number)
  random          enroll and test drawn at random, ignoring sessions (optimistic)
Accuracy is balanced (mean per-meaning recall), so chance is 1 / number of meanings.
"""
import csv
import sys
from pathlib import Path

import numpy as np

from risk_test import BASELINES, unit

TAKES = 5            # what the app asks for
MIN_CLIPS = 20       # a meaning needs this many clips to be scored
TEST_PER_CLASS = 15  # test clips per meaning per trial, so frequent meanings don't dominate
TRIALS = 50


def rows(root):
    with open(root / "dataset_file_directory.csv") as f:
        return [r for r in csv.DictReader(f) if (root / r["Filename"]).exists()]


def cache_path(root, model):
    return root / "embeddings" / (model.split("/")[-1] + ".npy")


def embed_all(root):
    from tink import audio
    from tink.embed import layers

    files = [r["Filename"] for r in rows(root)]
    for model in BASELINES:
        out = cache_path(root, model)
        if out.exists():
            continue
        out.parent.mkdir(exist_ok=True)
        H = []
        for i, name in enumerate(files):
            H.append(layers(audio.load(root / name), model).astype(np.float16))
            if i % 500 == 0:
                print(f"{model} {i}/{len(files)}", flush=True)
        np.save(out, np.stack(H))
        print(f"{model} done", flush=True)


def session(name):
    return name[:11]  # "200126_2142": recording date and time


def enrollable(idx, sess):
    """Sessions with enough clips of this meaning to enroll from, leaving enough in other sessions to test."""
    return sorted(s for s in set(sess[idx])
                  if (sess[idx] == s).sum() >= TAKES and (sess[idx] != s).sum() >= TEST_PER_CLASS)


def trial(X, labels, sess, rng, by_session):
    """One enroll/test split for one person. Returns balanced accuracy."""
    classes = sorted(set(labels))
    enroll, test = {}, {}
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if by_session:
            s = rng.choice(enrollable(idx, sess))
            enroll[c] = rng.choice(idx[sess[idx] == s], TAKES, replace=False)
            pool = idx[sess[idx] != s]
        else:
            perm = rng.permutation(idx)
            enroll[c], pool = perm[:TAKES], perm[TAKES:]
        test[c] = rng.choice(pool, min(TEST_PER_CLASS, len(pool)), replace=False)
    mu = X[np.concatenate(list(enroll.values()))].mean(axis=0)
    cents = unit(np.stack([X[enroll[c]].mean(axis=0) - mu for c in classes]))
    recalls = [np.mean((unit(X[test[c]] - mu) @ cents.T).argmax(axis=1) == k) for k, c in enumerate(classes)]
    return float(np.mean(recalls))


def people(root, by_session):
    """Per person: indices of the meanings that can be scored under this split."""
    rs = rows(root)
    labels = np.array([r["Label"] for r in rs])
    parts = np.array([r["Participant"] for r in rs])
    sess = np.array([session(r["Filename"]) for r in rs])
    groups = {}
    for p in sorted(set(parts)):
        keep = [idx for c in sorted(set(labels[parts == p]))
                if len(idx := np.flatnonzero((parts == p) & (labels == c))) >= MIN_CLIPS
                and (not by_session or enrollable(idx, sess))]
        if len(keep) >= 2:
            groups[p] = np.concatenate(keep)
    return groups, labels, sess


def score_layer(H, L, groups, labels, sess, by_session):
    """Mean balanced accuracy per person at one layer. Same seed for every model and layer."""
    X = H[:, L].astype(np.float32)
    res = {}
    for p, idx in groups.items():
        rng = np.random.default_rng(int(p[1:]))
        res[p] = float(np.mean([trial(X[idx], labels[idx], sess[idx], rng, by_session) for _ in range(TRIALS)]))
    return res


def score(root):
    for by_session, name in ((True, "other sessions"), (False, "random")):
        groups, labels, sess = people(root, by_session)
        chance = {p: 1 / len(set(labels[idx])) for p, idx in groups.items()}
        print(f"\n== Split: {name} ({TAKES} examples per meaning, {TRIALS} trials, balanced accuracy)")
        print("   " + "  ".join(f"{p}: {len(set(labels[idx]))} meanings" for p, idx in groups.items()))
        print(f"   chance (mean over people) {np.mean(list(chance.values())):.0%}\n")
        print(f"{'model':40} {'last layer':>10} {'layer picked on other people':>29}")
        for model in BASELINES:
            H = np.load(cache_path(root, model))
            per_layer = [score_layer(H, L, groups, labels, sess, by_session) for L in range(H.shape[1])]
            last = np.mean(list(per_layer[-1].values()))
            # leave one person out: pick the layer on the other people, score it on this one
            picked = np.mean([per_layer[int(np.argmax([np.mean([r[q] for q in r if q != p]) for r in per_layer]))][p]
                              for p in groups])
            print(f"{model:40} {last:>10.0%} {picked:>29.0%}")
            if model == BASELINES[0]:
                print("   per person, last layer:  " + "  ".join(
                    f"{p} {per_layer[-1][p]:.0%} (chance {chance[p]:.0%})" for p in groups))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("embed", "score"):
        sys.exit(__doc__)
    {"embed": embed_all, "score": score}[sys.argv[1]](Path(sys.argv[2]))
