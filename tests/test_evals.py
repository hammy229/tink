"""A benchmark that leaks or miscounts is worse than none, so the evaluation code gets tested too."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from evals import fewshot, metrics
from evals.datasets import Dataset, is_dysarthric, normalize_prompt


def blobs(sep: float, n_classes: int = 5, per_class: int = 30, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_classes, dim))
    X = np.concatenate([means[c] * sep + rng.normal(size=(per_class, dim)) for c in range(n_classes)])
    y = np.repeat(np.array([f"c{c}" for c in range(n_classes)]), per_class)
    sess = np.tile(np.repeat([f"s{i}" for i in range(3)], per_class // 3), n_classes)
    return X.astype(np.float32), y, sess


@pytest.mark.parametrize(
    ("text", "want"),
    [("stick", "stick"), ("tear [as in tear up that paper]", "tear"), ("knew", "know"), ("no", "know"),
     ("The quick brown fox.", None), ("xxx", None), ("[say Ah-P-Eee repeatedly]", None), ("Yes.", "yes")],
)
def test_normalize_prompt(text, want):
    assert normalize_prompt(text) == want


def test_controls_are_told_apart_from_dysarthric_speakers():
    assert is_dysarthric("F01") and is_dysarthric("M05")
    assert not is_dysarthric("FC02") and not is_dysarthric("MC01")


def test_dataset_rejects_mismatched_columns():
    with pytest.raises(ValueError, match="lengths"):
        Dataset("x", Path(), ("a",), np.array(["l", "m"]), np.array(["p"]), np.array(["s"]))


def test_across_sessions_never_tests_on_the_enrollment_session():
    _, y, sess = blobs(1.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        ep = fewshot.across_sessions(y, sess, rng)
        for enroll, test in zip(ep.enroll, ep.test, strict=True):
            assert len(set(sess[enroll])) == 1
            assert not set(sess[enroll]) & set(sess[test])
            assert len(enroll) == fewshot.RECANVO_TAKES


def test_word_episode_never_tests_on_an_enrolled_take():
    _, y, sess = blobs(1.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        ep = fewshot.word_episode(y, sess, rng)
        assert len(ep.enroll) == fewshot.WORD_WAYS
        for enroll, test in zip(ep.enroll, ep.test, strict=True):
            assert len(enroll) == fewshot.WORD_SHOTS and not set(enroll) & set(test)
            assert len(set(y[np.concatenate([enroll, test])])) == 1


def test_other_session_words_never_test_in_the_enrollment_session():
    _, y, sess = blobs(1.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        ep = fewshot.word_episode_across_sessions(y, sess, rng)
        assert len(ep.enroll) == fewshot.WORD_WAYS
        for enroll, test in zip(ep.enroll, ep.test, strict=True):
            assert len(enroll) == fewshot.WORD_SHOTS and len(set(sess[enroll])) == 1
            assert not set(sess[enroll]) & set(sess[test])


def test_every_word_in_an_other_session_episode_is_taught_in_one_sitting():
    _, y, sess = blobs(1.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        ep = fewshot.word_episode_across_sessions(y, sess, rng)
        assert len({sess[i] for enroll in ep.enroll for i in enroll}) == 1


def test_word_needs_takes_in_two_sessions_to_count():
    labels = np.array(["a", "a", "a", "b", "b", "b", "c", "c"])
    sessions = np.array(["s1", "s1", "s2", "s1", "s1", "s1", "s1", "s2"])
    assert fewshot.session_word_options(labels, sessions) == {"a": ["s1"]}


def test_episodes_are_reproducible():
    _, y, sess = blobs(1.0)
    a = fewshot.word_episode(y, sess, np.random.default_rng(3))
    b = fewshot.word_episode(y, sess, np.random.default_rng(3))
    assert all(np.array_equal(p, q) for p, q in zip(a.test, b.test, strict=True))


@pytest.mark.parametrize("method", list(fewshot.METHODS.values()))
def test_every_method_separates_easy_data(method):
    X, y, sess = blobs(3.0)
    r = fewshot.run_person(X, y, sess, method, fewshot.across_sessions, 0, 10)
    assert r["accuracy"] > 0.9


def _drifting(rng):
    """Every sound moves away from its first session, and stays moved on later days."""
    means = rng.normal(size=(5, 16))
    X, y, sess = [], [], []
    for c in range(5):
        moved = means[c] * 0.6 + rng.normal(size=16) * 1.6
        for s, center in (("s0", means[c] * 0.6), ("s1", moved), ("s2", moved)):
            X.append(center + rng.normal(size=(20, 16)))
            y += [f"c{c}"] * 20
            sess += [s] * 20
    return np.concatenate(X).astype(np.float32), np.array(y), np.array(sess)


def test_learning_from_confirmed_sounds_helps_when_sounds_really_drift():
    X, y, sess = _drifting(np.random.default_rng(0))

    def taught_on_the_first_day(labels, sessions, rng):
        enroll, test = [], []
        for c in sorted(set(labels)):
            idx = np.flatnonzero(labels == c)
            enroll.append(rng.choice(idx[sessions[idx] == "s0"], 5, replace=False))
            test.append(idx[sessions[idx] != "s0"])
        return fewshot.Episode(tuple(enroll), tuple(test))

    order = np.array([int(s[1]) * 1000 + i % 1000 for i, s in enumerate(sess)])
    r = fewshot.run_adaptive(X, y, sess, taught_on_the_first_day, 0, 20, recorded_at=order)
    assert r["adaptive"] > r["static"] + 0.02
    assert r["confirmed_clips_per_episode"] > 0


def test_a_careless_caregiver_adds_wrong_clips():
    X, y, sess = _drifting(np.random.default_rng(0))
    assert fewshot.run_adaptive(X, y, sess, fewshot.across_sessions, 0, 10, wrong_confirm=0.0, recorded_at=np.arange(len(y)))["wrong_clips_per_episode"] == 0
    assert fewshot.run_adaptive(X, y, sess, fewshot.across_sessions, 0, 10, wrong_confirm=0.9, recorded_at=np.arange(len(y)))["wrong_clips_per_episode"] > 0


def test_learning_runs_within_a_single_session():
    labels = np.repeat([f"c{i}" for i in range(5)], 6)
    X = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
    r = fewshot.run_adaptive(X, labels, np.full(30, "s0"), fewshot.word_episode, 0, 5, recorded_at=np.arange(30))
    assert r is not None and r["episodes"] == 5


def test_learning_confirms_earlier_clips_and_scores_later_ones():
    """Reversing the clock must change the result, or time order isn't really being used."""
    X, y, sess = _drifting(np.random.default_rng(0))
    forward = np.array([int(s[1]) * 1000 + i % 1000 for i, s in enumerate(sess)])
    a = fewshot.run_adaptive(X, y, sess, fewshot.across_sessions, 0, 10, recorded_at=forward)
    b = fewshot.run_adaptive(X, y, sess, fewshot.across_sessions, 0, 10, recorded_at=-forward)
    assert a != b


def test_one_sitting_episode_uses_one_session_and_tests_elsewhere():
    _, y, sess = blobs(1.0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        ep = fewshot.across_sessions_one_sitting(y, sess, rng)
        assert len(ep.enroll) >= 2 and len({sess[i] for e in ep.enroll for i in e}) == 1
        assert not {sess[i] for e in ep.enroll for i in e} & {sess[i] for t in ep.test for i in t}


def test_sign_flip_test_is_exact():
    assert metrics.sign_flip_p(np.array([1.0, 1.0, 1.0])) == pytest.approx(2 / 8)
    assert metrics.sign_flip_p(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])) == pytest.approx(2 / 64)
    assert metrics.sign_flip_p(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(1.0)


def test_six_people_all_better_is_enough_but_five_is_not():
    base = np.zeros(6)
    assert metrics.beats(base + np.arange(1, 7) / 100, base)["better"] is True
    assert metrics.beats(np.zeros(5) + np.arange(1, 6) / 100, np.zeros(5))["better"] is False


def test_shuffled_labels_score_at_chance():
    X, y, sess = blobs(3.0)
    shuffled = np.random.default_rng(0).permutation(y)
    r = fewshot.run_person(X, shuffled, sess, fewshot.centroid, fewshot.word_episode, 0, 300)
    assert r["accuracy"] == pytest.approx(r["chance"], abs=0.07)


def test_ranks_counts_questions_needed():
    scores = np.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]])
    assert metrics.ranks(scores, np.array([0, 1, 2])).tolist() == [1, 2, 3]


def test_balanced_accuracy_is_not_fooled_by_a_common_class():
    scores = np.tile([1.0, 0.0], (100, 1))
    y = np.array([0] * 99 + [1])
    assert metrics.balanced_accuracy(scores, y, 2) == 0.5


def test_ece_is_zero_when_confidence_matches_accuracy_and_large_when_it_lies():
    correct = np.array([True] * 8 + [False] * 2)
    assert metrics.expected_calibration_error(np.full(10, 0.8), correct) == pytest.approx(0)
    assert metrics.expected_calibration_error(np.full(10, 0.99), np.zeros(10, bool)) == pytest.approx(0.99)


def test_risk_coverage_answers_only_on_the_most_confident():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    ok = np.array([True, True, False, False])
    rc = metrics.risk_coverage(conf, ok, coverages=(1.0, 0.5))
    assert rc == {"1": 0.5, "0.5": 1.0}


def test_conformal_threshold_delivers_its_promise_on_exchangeable_data():
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(500):
        scores = rng.random(40)
        q = metrics.conformal_threshold(scores[:30], alpha=0.1)
        covered.extend((1 - scores[30:]) <= q)
    assert np.mean(covered) >= 0.88


def test_bootstrap_interval_contains_the_mean_and_narrows_with_people():
    rng = np.random.default_rng(0)
    few = metrics.bootstrap_ci(rng.normal(0.3, 0.05, 8))
    many = metrics.bootstrap_ci(rng.normal(0.3, 0.05, 800))
    assert few["low"] < few["mean"] < few["high"]
    assert many["high"] - many["low"] < few["high"] - few["low"]


def test_beats_needs_the_whole_interval_above_zero():
    base = np.array([0.2, 0.3, 0.25, 0.4, 0.3, 0.35])
    assert metrics.beats(base + np.array([0.1, 0.12, 0.09, 0.11, 0.1, 0.13]), base)["better"] is True
    assert metrics.beats(base + np.array([0.1, -0.1, 0.05, -0.05, 0.02, -0.02]), base)["better"] is False


def test_logmel_baseline_shape_and_normalization():
    from evals.embed import logmel_stats

    out = logmel_stats(np.random.default_rng(0).normal(size=16_000).astype(np.float32))
    assert out.shape == (1, 80) and np.linalg.norm(out) == pytest.approx(1, abs=1e-5)
    assert logmel_stats(np.zeros(10, np.float32)).shape == (1, 80)  # very short clips don't crash


def test_head_learns_a_task_and_never_sees_the_held_out_person():
    torch = pytest.importorskip("torch")
    from evals import head

    tasks = []
    for seed in range(4):
        X, y, sess = blobs(2.0, seed=seed)
        tasks.append(head.Task(torch.from_numpy(X), y, sess, fewshot.word_episode))
    cfg = head.HeadConfig(episodes=200, eval_every=50, val_episodes=5, dim=8, seed=0)
    h, info = head.train_head(tasks[:3], tasks[3:], cfg)
    assert info["val_accuracy"] > 0.7
    X, y, sess = blobs(2.0, seed=99)
    r = fewshot.run_person(head.apply(h, X), y, sess, fewshot.centroid, fewshot.word_episode, 0, 20)
    assert r["accuracy"] > 0.6


def test_head_training_is_deterministic():
    torch = pytest.importorskip("torch")
    from evals import head

    X, y, sess = blobs(2.0)
    task = head.Task(torch.from_numpy(X), y, sess, fewshot.word_episode)
    cfg = head.HeadConfig(episodes=20, eval_every=10, val_episodes=2, dim=4, seed=7)
    a, _ = head.train_head([task], [task], cfg)
    b, _ = head.train_head([task], [task], cfg)
    assert torch.equal(a.net[-1].weight, b.net[-1].weight)


def test_the_pinned_browser_model_has_a_passing_gate_report():
    """A build can't be pinned for the site unless the benchmarks were re-scored with that exact file and passed."""
    import json
    import re

    root = Path(__file__).parent.parent
    config = (root / "web" / "engine" / "config.js").read_text()
    pinned = re.search(r'sha256: "([0-9a-f]{64})"', config).group(1)
    reports = [json.loads(p.read_text()) for p in (root / "evals" / "results").glob("gate-*.json")]
    matching = [r for r in reports if r.get("sha256") == pinned]
    assert matching, "run evals/onnx_gate.py on the pinned model"
    assert all(r["passes"] for r in matching)


def test_published_numbers_match_the_raw_results():
    from evals import render

    report, load, gates, routing = render.load_all()
    assert render.DOC.read_text() == render.render(report, load, gates, routing), "run: python -m evals.render"
    readme = (render.ROOT / "README.md").read_text()
    assert render.with_readme_block(readme, report) == readme, "run: python -m evals.render"
