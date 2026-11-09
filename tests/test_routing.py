"""The routing benchmark decides how much traffic goes to a server, so its bookkeeping gets tested too."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from evals import fewshot, routing
from tests.test_evals import blobs


def test_route_sends_only_sounds_below_the_threshold():
    device, server = np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])
    conf = np.array([0.9, 0.5, 0.7, 0.2])
    pred, sent = routing.route(device, server, conf, 0.6)
    assert sent.tolist() == [False, True, False, True]
    assert pred.tolist() == [0, 1, 0, 1]


def test_threshold_zero_never_routes_and_above_one_always_routes():
    device, server, conf = np.array([0, 1]), np.array([1, 0]), np.array([0.3, 1.0])
    assert routing.route(device, server, conf, 0.0)[1].sum() == 0
    assert routing.route(device, server, conf, routing.ALWAYS)[1].all()


def test_balanced_accuracy_from_predictions():
    y = np.array([0] * 99 + [1])
    assert routing.balanced(np.zeros(100, int), y, 2) == 0.5


def _trial(conf, device_right, server_right):
    y = np.arange(len(conf)) % 2
    return routing.Trial(device=np.where(device_right, y, 1 - y), server=np.where(server_right, y, 1 - y),
                         confidence=np.asarray(conf, float), truth=y, ways=2)


def test_confident_routing_beats_random_routing_when_confidence_is_informative():
    # the device is right exactly when it's sure, and the server is always right
    conf = np.linspace(0.05, 0.95, 40)
    trial = _trial(conf, conf >= 0.5, np.ones(40, bool))
    r = routing.score_person([trial], thresholds=(0.5,), seed=0)["0.5"]
    assert r["fallback"] == pytest.approx(0.5)
    assert r["accuracy"] == 1.0
    assert r["random"] < r["accuracy"]


def test_routing_more_never_hurts_when_the_server_is_always_right():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    trial = _trial(conf, rng.random(200) < conf, np.ones(200, bool))
    r = routing.score_person([trial], thresholds=(0.0, 0.3, 0.6, 0.9, routing.ALWAYS), seed=0)
    accs = [r[k]["accuracy"] for k in r]
    rates = [r[k]["fallback"] for k in r]
    assert accs == sorted(accs) and rates == sorted(rates)
    assert rates[0] == 0 and rates[-1] == 1 and accs[-1] == 1.0


def test_budget_threshold_is_picked_from_other_people_only():
    rates = {"A": {"0.3": 0.1, "0.6": 0.2, "0.9": 0.5},
             "B": {"0.3": 0.1, "0.6": 0.3, "0.9": 0.6},
             "C": {"0.3": 0.9, "0.6": 0.9, "0.9": 0.9}}
    assert routing.pick_threshold(rates, held_out="C", budget=0.3) == "0.6"
    changed = {**rates, "C": {"0.3": 0.0, "0.6": 0.0, "0.9": 0.0}}
    assert routing.pick_threshold(changed, held_out="C", budget=0.3) == "0.6"
    assert routing.pick_threshold(rates, held_out="A", budget=0.05) == "0"


def test_person_trials_use_the_same_episodes_for_device_and_server():
    X, y, sess = blobs(3.0)
    noisy = X + np.random.default_rng(1).normal(scale=5.0, size=X.shape).astype(np.float32)
    trials = routing.person_trials(noisy, X, y, sess, fewshot.word_episode, seed=0, trials=20)
    assert len(trials) == 20
    for t in trials:
        assert len(t.device) == len(t.server) == len(t.truth) == len(t.confidence)
        assert ((t.confidence > 0) & (t.confidence <= 1)).all()
    device = np.mean([routing.balanced(t.device, t.truth, t.ways) for t in trials])
    server = np.mean([routing.balanced(t.server, t.truth, t.ways) for t in trials])
    assert server > device


def test_summary_reports_the_share_of_the_server_gain_and_none_when_the_server_is_worse():
    def person(device, server, rate):
        return {"0": {"accuracy": device, "fallback": 0.0, "random": device},
                "0.5": {"accuracy": (device + server) / 2, "fallback": rate, "random": device},
                "1.01": {"accuracy": server, "fallback": 1.0, "random": server}}

    better = routing.summarize({p: person(0.6, 0.8, 0.2) for p in ("A", "B", "C")})["budget"]
    assert better["threshold_per_person"] == {"A": "0.5", "B": "0.5", "C": "0.5"}
    assert better["share_of_server_gain"] == pytest.approx(0.5)
    worse = routing.summarize({p: person(0.6, 0.5, 0.2) for p in ("A", "B", "C")})["budget"]
    assert worse["share_of_server_gain"] is None
