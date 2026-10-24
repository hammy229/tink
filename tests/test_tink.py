"""Tests for the parts the live demo depends on. Uses synthetic tones, so no recordings are needed.

    uv run pytest -q
"""
import io

import numpy as np
import pytest
import soundfile as sf

from tink import bayes, llm


def tone(freq: float, seconds: float = 0.8, sr: int = 16_000, noise: float = 0.003, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    wav = 0.5 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t)) + rng.normal(0, noise, len(t))
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def hiss(seconds: float = 0.8, sr: int = 16_000, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    wav = rng.normal(0, 0.3, int(sr * seconds)) * np.hanning(int(sr * seconds))
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


# ---------- bayes ----------

def test_weak_match_loses_to_unknown():
    ranked = bayes.posterior({"Hum": 0.1, "Sigh": 0.05}, 0.5, {}, {}, [], 14)["ranked"]
    assert ranked[0]["name"] == bayes.UNKNOWN


def test_face_cue_breaks_a_tie():
    scores, cues = {"Hum": 0.55, "Sigh": 0.53}, {"Hum": ["smile"], "Sigh": ["eyes_closed"]}
    assert bayes.posterior(scores, 0.5, {}, cues, [], 14)["ranked"][0]["name"] == "Hum"
    assert bayes.posterior(scores, 0.5, {}, cues, ["eyes_closed"], 14)["ranked"][0]["name"] == "Sigh"


def test_time_of_day_prior_learns_usage():
    ranked = bayes.posterior({"Hum": 0.55, "Sigh": 0.53}, 0.5, {"Sigh": {"afternoon": 6}}, {}, [], 14)["ranked"]
    assert ranked[0]["name"] == "Sigh"


def test_probabilities_sum_to_one_and_ignore_unknown_cues():
    out = bayes.posterior({"A": 0.7, "B": 0.2}, 0.5, {}, {}, ["smile", "not-a-cue"], 9)
    assert out["cues"] == ["smile"]
    assert sum(r["p"] for r in out["ranked"]) == pytest.approx(1, abs=0.01)


# ---------- llm parsing ----------

def test_parse_strips_fences_and_is_strict_about_ask():
    assert llm._parse('```json\n{"say": "Water, please.", "ask": false}\n```') == {"say": "Water, please.", "ask": False}
    assert llm._parse('{"say": "hi", "ask": "false"}')["ask"] is False  # a string is not true
    assert llm._parse('{"say": "  "}') is None


# ---------- store + API ----------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TINK_DATA", str(tmp_path))
    import importlib

    from tink import app as app_module, store as store_module
    importlib.reload(store_module)
    importlib.reload(app_module)
    monkeypatch.setattr(app_module.llm, "KEY", None)  # never call Grok from tests
    from fastapi.testclient import TestClient
    return TestClient(app_module.app), app_module, tmp_path


def enroll(c, name, clip, phrase="x"):
    return c.post("/api/enroll", data={"name": name, "phrase": phrase}, files={"audio": ("a.wav", clip)})


def test_needs_two_sounds_before_matching(client):
    c, _, _ = client
    enroll(c, "Hum", tone(140))
    r = c.post("/api/recognize", files={"audio": ("a.wav", tone(140))})
    assert r.status_code == 409


def test_similar_names_never_share_a_folder(client):
    c, app_module, root = client
    for i in range(2):
        enroll(c, "Hum", tone(140, seed=i))
        enroll(c, "hum!", hiss(seed=i))
    dirs = {s["dir"] for s in app_module.store.sounds.values()}
    assert len(dirs) == 2
    assert c.delete("/api/sounds/hum!").json() == {"ok": True}
    assert all((root / "takes" / t).exists() for t in app_module.store.sounds["Hum"]["takes"])


def test_recognize_end_to_end_with_fallback(client):
    c, _, _ = client
    for i in range(3):
        enroll(c, "Hum", tone(140, seed=i), "I'm thirsty.")
        enroll(c, "Hiss", hiss(seed=i), "I'm cold.")
    r = c.post("/api/recognize", data={"speak_in": "Spanish"}, files={"audio": ("a.wav", tone(145, seed=9))}).json()
    assert r["name"] == "Hum"
    assert r["source"] == "phrase" and r["say"] == "I'm thirsty."
    assert r["packet"]["speak_in"] == "Spanish"
    assert {"ranked", "scores", "taught"} <= r.keys()


def test_rejects_silence_and_long_text(client):
    c, _, _ = client
    silent = io.BytesIO()
    sf.write(silent, np.zeros(16_000, np.float32), 16_000, format="WAV")
    assert enroll(c, "x", silent.getvalue()).status_code == 400
    assert enroll(c, "x" * 90, tone(140)).status_code == 400


def test_names_with_slashes_can_be_forgotten(client):
    c, _, _ = client
    enroll(c, "AC/DC", tone(300))
    assert c.delete("/api/sounds/AC/DC").status_code == 200


def test_store_survives_a_lost_take(client):
    c, app_module, root = client
    enroll(c, "Hum", tone(140))
    enroll(c, "Hum", tone(141, seed=1))
    (root / "takes" / app_module.store.sounds["Hum"]["takes"][0]).unlink()
    reloaded = app_module.Store(root)
    assert len(reloaded.sounds["Hum"]["takes"]) == 1


def test_yes_no_dialogue_resolves_an_unsure_guess(client, monkeypatch):
    c, app_module, _ = client
    monkeypatch.setattr(app_module, "ASK_BELOW", 1.01)  # make every ordinary match a question
    for i in range(3):
        enroll(c, "Hum", tone(140, seed=i), "I'm thirsty.")
        enroll(c, "Hiss", hiss(seed=i), "I'm cold.")
        c.post("/api/enroll", data={"name": "Ah", "phrase": "Yes", "role": "yes"}, files={"audio": ("a.wav", tone(700, seed=i))})
        c.post("/api/enroll", data={"name": "Mm", "phrase": "No", "role": "no"}, files={"audio": ("a.wav", tone(1800, seed=i))})
    ask = c.post("/api/recognize", files={"audio": ("a.wav", tone(142, seed=7))}).json()
    assert ask["ask"] and ask["dialog"] == {"asking": "Hum"} and "I'm thirsty." in ask["say"]

    no = c.post("/api/recognize", files={"audio": ("a.wav", tone(1800, seed=8))}).json()
    assert no["dialog"]["answered"] == "no" and no["ask"]

    # ask again from scratch, then answer yes
    c.post("/api/recognize", files={"audio": ("a.wav", tone(142, seed=9))})
    yes = c.post("/api/recognize", files={"audio": ("a.wav", tone(700, seed=10))}).json()
    assert yes["dialog"] == {"answered": "yes", "meant": "Hum"}
    assert yes["say"] == "I'm thirsty." and yes["ask"] is False and yes["name"] == "Hum"


def test_only_one_sound_holds_each_role(client):
    c, app_module, _ = client
    c.post("/api/enroll", data={"name": "Ah", "phrase": "Yes", "role": "yes"}, files={"audio": ("a.wav", tone(700))})
    c.post("/api/enroll", data={"name": "Oh", "phrase": "Yes", "role": "yes"}, files={"audio": ("a.wav", tone(900))})
    assert app_module.store.role_of("Ah") is None and app_module.store.role_of("Oh") == "yes"


def test_room_noise_keeps_tink_quiet(client):
    c, _, _ = client
    for i in range(3):
        enroll(c, "Hum", tone(140, seed=i), "I'm thirsty.")
        enroll(c, "Hiss", hiss(seed=i), "I'm cold.")
        c.post("/api/enroll", data={"name": "Room noise", "phrase": "(room noise)", "role": "background"},
               files={"audio": ("a.wav", tone(3000, seed=i))})
    r = c.post("/api/recognize", files={"audio": ("a.wav", tone(3000, seed=9))}).json()
    assert r["background"] is True and r["say"] == "" and r["matched"] is False


def test_quiet_room_is_a_valid_background_take(client):
    c, _, _ = client
    silent = io.BytesIO()
    sf.write(silent, np.zeros(16_000, np.float32), 16_000, format="WAV")
    r = c.post("/api/enroll", data={"name": "Room noise", "phrase": "(room noise)", "role": "background"},
               files={"audio": ("a.wav", silent.getvalue())})
    assert r.status_code == 200


def test_urgent_sound_alerts_and_skips_the_question(client, monkeypatch):
    c, app_module, _ = client
    monkeypatch.setattr(app_module, "ASK_BELOW", 1.01)  # an ordinary match would become a question
    for i in range(3):
        enroll(c, "Hum", tone(140, seed=i), "I'm thirsty.")
        c.post("/api/enroll", data={"name": "Scream", "phrase": "Help, it hurts!", "urgent": "true"},
               files={"audio": ("a.wav", hiss(seed=i))})
        c.post("/api/enroll", data={"name": "Ah", "phrase": "Yes", "role": "yes"}, files={"audio": ("a.wav", tone(700, seed=i))})
        c.post("/api/enroll", data={"name": "Mm", "phrase": "No", "role": "no"}, files={"audio": ("a.wav", tone(1800, seed=i))})
    r = c.post("/api/recognize", files={"audio": ("a.wav", hiss(seed=9))}).json()
    assert r["urgent"] is True and r["ask"] is False and r["say"] == "Help, it hurts!" and r["source"] == "urgent"
    assert any(s["urgent"] for s in c.get("/api/sounds").json()["sounds"])


def test_reply_says_it_as_typed_without_grok(client):
    c, _, _ = client
    r = c.post("/api/reply", data={"text": "I'm coming, two minutes.", "to": "Spanish"}).json()
    assert r["say"] == "I'm coming, two minutes." and r["source"] == "typed" and r["to"] == "Spanish"
    assert c.post("/api/reply", data={"text": "   "}).status_code == 400
    assert c.post("/api/reply", data={"text": "x" * 201}).status_code == 400


def test_reply_uses_grok_translation_when_available(client, monkeypatch):
    c, app_module, _ = client

    async def fake_translate(text, to):
        return "Ya voy, dos minutos."
    monkeypatch.setattr(app_module.llm, "translate", fake_translate)
    r = c.post("/api/reply", data={"text": "I'm coming, two minutes.", "to": "Spanish"}).json()
    assert r["say"] == "Ya voy, dos minutos." and r["source"] == "grok" and r["typed"] == "I'm coming, two minutes."
