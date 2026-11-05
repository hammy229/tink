"""Failures that would hurt a real person: lost profiles, memory blowups, slow restarts."""
import io
import json

import numpy as np
import pytest
import soundfile as sf

from tink import audio
from tink.store import Store, atomic_write


def wav_bytes(seconds: float, fmt: str = "FLAC") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16_000 * seconds), np.float32), 16_000, format=fmt)
    return buf.getvalue()


def test_small_upload_that_decodes_to_hours_is_refused_before_decoding():
    bomb = wav_bytes(600)
    assert len(bomb) < 2 * 1024 * 1024  # small on the wire, so the byte limit alone can't catch it
    with pytest.raises(audio.ClipTooLong):
        audio.load_upload(bomb)


def test_normal_upload_still_loads():
    assert len(audio.load_upload(wav_bytes(1, "WAV"))) > 0


def test_crash_during_write_leaves_the_old_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "sounds.json"
    atomic_write(path, b'{"good": 1}')

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr("tink.store.Path.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write(path, b'{"half')
    assert json.loads(path.read_text()) == {"good": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["sounds.json"]  # no temp files left behind


def _fake_embed(calls):
    def embed(wav):
        calls.append(len(wav))
        return np.ones(4, np.float32) / 2

    return embed


TONE = np.sin(np.linspace(0, 400, 16_000)).astype(np.float32)


def test_restart_reuses_cached_embeddings(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr("tink.store.embed", _fake_embed(calls))
    first = Store(tmp_path)
    first.add_take("Hum", "x", TONE)
    assert calls  # embedded on the way in

    calls.clear()
    second = Store(tmp_path)
    assert not calls  # the takes come from the cache
    assert np.array_equal(second.vecs["Hum"][0], first.vecs["Hum"][0])


def test_corrupt_cache_is_rebuilt_not_fatal(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr("tink.store.embed", _fake_embed(calls))
    Store(tmp_path).add_take("Hum", "x", TONE)
    for cached in (tmp_path / "takes").rglob("*.npy"):
        cached.write_bytes(b"not numpy")
    calls.clear()
    assert len(Store(tmp_path).vecs["Hum"]) == 1
    assert calls


def test_forgetting_a_sound_removes_its_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("tink.store.embed", _fake_embed([]))
    store = Store(tmp_path)
    store.add_take("Hum", "x", TONE)
    store.delete("Hum")
    assert not list((tmp_path / "takes").rglob("*"))


def test_changing_the_model_invalidates_the_cache(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr("tink.store.embed", _fake_embed(calls))
    Store(tmp_path).add_take("Hum", "x", TONE)
    calls.clear()
    monkeypatch.setattr("tink.store.cache_tag", lambda: "another-model")
    Store(tmp_path)
    assert len(calls) >= 5  # the take and its 4 variants again
