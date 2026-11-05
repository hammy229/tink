"""Few-shot evaluation: enroll a handful of takes per sound, then recognize new ones, the way the app is used."""
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from evals import metrics

RECANVO_TAKES = 5  # what the app asks for
RECANVO_MIN_CLIPS = 20  # a meaning needs this many clips to be scored
RECANVO_TEST_PER_CLASS = 15  # so frequent meanings don't dominate
WORD_WAYS, WORD_SHOTS = 5, 2
LDA_SHRINKAGE = 0.5


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def person_seed(person: str) -> int:
    tail = person[1:]
    return int(tail) if tail.isdigit() else zlib.crc32(person.encode())


# ---------- episodes ----------

@dataclass(frozen=True)
class Episode:
    enroll: tuple[np.ndarray, ...]  # per class: indices to learn from
    test: tuple[np.ndarray, ...]  # per class: indices to recognize


EpisodeFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], Episode]


def _enrollable(idx: np.ndarray, sessions: np.ndarray) -> list:
    """Sessions with enough clips of this meaning to enroll from, leaving enough in other sessions to test."""
    return sorted(s for s in set(sessions[idx])
                  if (sessions[idx] == s).sum() >= RECANVO_TAKES
                  and (sessions[idx] != s).sum() >= RECANVO_TEST_PER_CLASS)


def across_sessions(labels: np.ndarray, sessions: np.ndarray, rng: np.random.Generator) -> Episode:
    """Teach once, use on another day: enroll from one session, test only on others."""
    enroll, test = [], []
    for c in sorted(set(labels)):
        idx = np.flatnonzero(labels == c)
        s = rng.choice(_enrollable(idx, sessions))
        enroll.append(rng.choice(idx[sessions[idx] == s], RECANVO_TAKES, replace=False))
        pool = idx[sessions[idx] != s]
        test.append(rng.choice(pool, min(RECANVO_TEST_PER_CLASS, len(pool)), replace=False))
    return Episode(tuple(enroll), tuple(test))


def word_episode(labels: np.ndarray, _sessions: np.ndarray, rng: np.random.Generator) -> Episode:
    """N words at random, K takes of each to learn, the remaining takes to test."""
    words = rng.choice(sorted(set(labels)), WORD_WAYS, replace=False)
    enroll, test = [], []
    for w in words:
        perm = rng.permutation(np.flatnonzero(labels == w))
        enroll.append(perm[:WORD_SHOTS])
        test.append(perm[WORD_SHOTS:])
    return Episode(tuple(enroll), tuple(test))


def session_word_options(labels: np.ndarray, sessions: np.ndarray) -> dict:
    """For each word, the sessions with enough takes to enroll from and at least one take elsewhere to test on."""
    options = {}
    for w in set(labels):
        idx = np.flatnonzero(labels == w)
        good = sorted(s for s in set(sessions[idx])
                      if (sessions[idx] == s).sum() >= WORD_SHOTS and (sessions[idx] != s).any())
        if good:
            options[w] = good
    return options


def enrollment_sessions(labels: np.ndarray, sessions: np.ndarray) -> dict:
    """Sessions where someone could teach a whole episode in one sitting, and the words they could teach there."""
    by_session: dict = {}
    for w, good in session_word_options(labels, sessions).items():
        for s in good:
            by_session.setdefault(s, []).append(w)
    return {s: sorted(ws) for s, ws in by_session.items() if len(ws) >= WORD_WAYS}


def word_episode_across_sessions(labels: np.ndarray, sessions: np.ndarray, rng: np.random.Generator) -> Episode:
    """Teach once, use on another day, with deliberate sounds. Every word is taught in the same sitting, as in the
    app, and tested only on other sessions, so a session can't stand in for a word."""
    choices = enrollment_sessions(labels, sessions)
    s = rng.choice(sorted(choices))
    enroll, test = [], []
    for w in rng.choice(choices[s], WORD_WAYS, replace=False):
        idx = np.flatnonzero(labels == w)
        enroll.append(rng.choice(idx[sessions[idx] == s], WORD_SHOTS, replace=False))
        test.append(idx[sessions[idx] != s])
    return Episode(tuple(enroll), tuple(test))


def usable_for_sessions(labels: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    """Clips of meanings with enough data to enroll in one session and test in another."""
    keep = np.zeros(len(labels), bool)
    for c in set(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) >= RECANVO_MIN_CLIPS and _enrollable(idx, sessions):
            keep[idx] = True
    return keep


# ---------- methods ----------

Method = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def centroid(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """What the app ships: subtract the mean of all takes, then cosine to each sound's average."""
    mu = X.mean(axis=0)
    cents = unit(np.stack([X[y == c].mean(axis=0) - mu for c in np.unique(y)]))
    return unit(Q - mu) @ cents.T


def uncentered(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Does centering earn its place? Plain cosine to each sound's average."""
    return unit(Q) @ unit(np.stack([X[y == c].mean(axis=0) for c in np.unique(y)])).T


def nearest(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Best single take per sound instead of the average. Better if one sound has several forms."""
    mu = X.mean(axis=0)
    sims = unit(Q - mu) @ unit(X - mu).T
    return np.stack([sims[:, y == c].max(axis=1) for c in np.unique(y)], axis=1)


def shrunk_lda(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Learn the person's own within-sound variation from their takes.

    Shrinkage is fixed at 0.5 up front, not tuned on these benchmarks. shrinkage="auto" is unusable here: with 10 clips
    in 768 dimensions it falls to chance, because Ledoit-Wolf on standardized features can't work with so few samples.
    """
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=LDA_SHRINKAGE).fit(X.astype(np.float64), y)
    scores = model.decision_function(Q.astype(np.float64))
    return scores if scores.ndim == 2 else np.stack([-scores, scores], axis=1)


METHODS: dict[str, Method] = {"centroid": centroid, "uncentered": uncentered, "nearest": nearest, "lda": shrunk_lda}


# ---------- running ----------

def run_person(X, labels, sessions, method: Method, episode: EpisodeFn, seed: int, trials: int,
               augmented: np.ndarray | None = None) -> dict:
    """Same seed for every method and model, so comparisons are paired on identical episodes.

    `augmented` is (clips, variants, dim). When given, enrollment also uses each take's variants, as the app does.
    """
    rng = np.random.default_rng(seed)
    accs, qranks, chance = [], [], []
    for _ in range(trials):
        ep = episode(labels, sessions, rng)
        Xe = np.concatenate([X[i] for i in ep.enroll])
        ye = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.enroll)])
        if augmented is not None:
            extra = [augmented[i].reshape(-1, X.shape[1]) for i in ep.enroll]
            Xe = np.concatenate([Xe, *extra])
            ye = np.concatenate([ye, *[np.full(len(e), k) for k, e in enumerate(extra)]])
        Xt = np.concatenate([X[i] for i in ep.test])
        yt = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.test)])
        scores = method(Xe, ye, Xt)
        accs.append(metrics.balanced_accuracy(scores, yt, len(ep.enroll)))
        qranks.append(metrics.ranks(scores, yt))
        chance.append(1 / len(ep.enroll))
    r = np.concatenate(qranks)
    return {"accuracy": float(np.mean(accs)), "chance": float(np.mean(chance)), "mean_questions": float(r.mean()),
            "within_2": float((r <= 2).mean()), "within_3": float((r <= 3).mean())}


MAX_LEARNED = 30  # same cap as tink.store.MAX_LEARNED_TAKES


def run_adaptive(X, labels, sessions, episode: EpisodeFn, seed: int, trials: int,
                 wrong_confirm: float = 0.1) -> dict | None:
    """Does learning from confirmed sounds help on later days? A caregiver's "Right" adds that clip to the sound.

    Time order, not random order: the test clips play session by session. Earlier sessions are where people confirm,
    and only the later sessions are scored, so a clip can't help a near copy of itself from the same sitting.
    Correct guesses get confirmed, and a tired caregiver also confirms a wrong guess `wrong_confirm` of the time, which
    adds that clip to the wrong sound. Returns None when this person has too few sessions to tell.
    """
    rng = np.random.default_rng(seed)
    static_acc, adaptive_acc, added, wrongly = [], [], [], []
    for _ in range(trials):
        ep = episode(labels, sessions, rng)
        Xe = np.concatenate([X[i] for i in ep.enroll])
        ye = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.enroll)])
        pool = np.concatenate(ep.test)
        truth_of = {int(i): k for k, idx in enumerate(ep.test) for i in idx}
        days = sorted(set(sessions[pool]))
        if len(days) < 2:
            continue
        cut = max(1, len(days) // 2)
        earlier = rng.permutation(pool[np.isin(sessions[pool], days[:cut])])
        later = pool[np.isin(sessions[pool], days[cut:])]
        order = sorted(earlier, key=lambda i: sessions[i])
        y_later = np.array([truth_of[int(i)] for i in later])
        if len(set(y_later)) < 2:
            continue
        grown_X, grown_y = Xe, ye
        bad = 0
        for i in order:
            guess = int(centroid(grown_X, grown_y, X[i:i + 1]).argmax())
            right = guess == truth_of[int(i)]
            if (right or rng.random() < wrong_confirm) and (grown_y == guess).sum() < MAX_LEARNED:
                grown_X = np.concatenate([grown_X, X[i:i + 1]])
                grown_y = np.append(grown_y, guess)
                bad += not right
        ways = len(ep.enroll)
        static_acc.append(metrics.balanced_accuracy(centroid(Xe, ye, X[later]), y_later, ways))
        adaptive_acc.append(metrics.balanced_accuracy(centroid(grown_X, grown_y, X[later]), y_later, ways))
        added.append(len(grown_y) - len(ye))
        wrongly.append(bad)
    if not static_acc:
        return None
    return {"static": float(np.mean(static_acc)), "adaptive": float(np.mean(adaptive_acc)),
            "confirmed_clips_per_episode": float(np.mean(added)), "wrong_clips_per_episode": float(np.mean(wrongly)),
            "episodes": len(static_acc)}


def _leave_one_take_out(Xe: np.ndarray, ye: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for i in range(len(ye)):
        keep = np.arange(len(ye)) != i
        rows.append(centroid(Xe[keep], ye[keep], Xe[i:i + 1])[0])
    return np.stack(rows), ye


TEMPS = np.geomspace(0.04, 1.0, 50)  # same grid and floor as tink.store


def _fit_temp(S: np.ndarray, y: np.ndarray) -> float:
    def nll(t):
        z = S / t
        z = z - z.max(axis=1, keepdims=True)
        return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean()

    return float(min(TEMPS, key=nll))


def run_uncertainty(X, labels, sessions, episode: EpisodeFn, seed: int, trials: int, alpha: float = 0.1) -> dict:
    """Can Tink trust its own confidence? Calibrates only on the enrolled takes, exactly as the app does."""
    rng = np.random.default_rng(seed)
    conf, conf_fixed, correct, covered, sizes, ways = [], [], [], [], [], []
    for _ in range(trials):
        ep = episode(labels, sessions, rng)
        Xe = np.concatenate([X[i] for i in ep.enroll])
        ye = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.enroll)])
        Xt = np.concatenate([X[i] for i in ep.test])
        yt = np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.test)])
        S = centroid(Xe, ye, Xt)
        loo, loo_y = _leave_one_take_out(Xe, ye)
        conf.append(metrics.softmax(S, _fit_temp(loo, loo_y)).max(axis=1))
        conf_fixed.append(metrics.softmax(S, 0.1).max(axis=1))
        correct.append(S.argmax(axis=1) == yt)
        q = metrics.conformal_threshold(loo[np.arange(len(loo_y)), loo_y], alpha)
        member = q >= (1 - S)
        covered.append(member[np.arange(len(yt)), yt])
        sizes.append(member.sum(axis=1))
        ways.append(len(ep.enroll))
    c, ok = np.concatenate(conf), np.concatenate(correct)
    return {
        "ece_self_calibrated": metrics.expected_calibration_error(c, ok),
        "ece_fixed_temp": metrics.expected_calibration_error(np.concatenate(conf_fixed), ok),
        "mean_confidence": float(c.mean()),
        "accuracy": float(ok.mean()),
        "risk_coverage": metrics.risk_coverage(c, ok),
        "conformal_target": 1 - alpha,
        "conformal_coverage": float(np.concatenate(covered).mean()),
        "conformal_mean_set": float(np.concatenate(sizes).mean()),
        "ways": float(np.mean(ways)),
    }
