"""Hands-free streaming over a WebSocket: audio in, one recognized turn out per sound."""
import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.helpers import hiss, tone


def _pcm(*parts: np.ndarray) -> bytes:
    return (np.concatenate(parts) * 32767).astype(np.int16).tobytes()


def _sine(freq: float, seconds: float = 0.8) -> np.ndarray:
    t = np.arange(int(16_000 * seconds)) / 16_000
    return 0.4 * np.sin(2 * np.pi * freq * t)


def send(ws, data: bytes, size: int = 8192) -> None:
    for i in range(0, len(data), size):
        ws.send_bytes(data[i : i + size])


QUIET = np.random.default_rng(0).normal(0, 0.0005, 16_000)


@pytest.fixture()
def streaming(tmp_path, monkeypatch):
    monkeypatch.setenv("TINK_DATA", str(tmp_path))
    from tink import app as app_module
    from tink import profiles as profiles_module
    from tink import store as store_module

    importlib.reload(store_module)
    importlib.reload(profiles_module)
    importlib.reload(app_module)
    monkeypatch.setattr(app_module.llm, "KEY", None)
    c = TestClient(app_module.app)
    for i in range(3):
        c.post("/api/enroll", data={"name": "Hum", "phrase": "I'm thirsty."}, files={"audio": ("a.wav", tone(140, seed=i))})
        c.post("/api/enroll", data={"name": "Hiss", "phrase": "I'm cold."}, files={"audio": ("a.wav", hiss(seed=i))})
    return c, app_module


def test_a_sound_in_the_stream_becomes_a_turn(streaming):
    c, _ = streaming
    with c.websocket_connect("/api/listen") as ws:
        send(ws, _pcm(QUIET, _sine(142), QUIET))
        out = ws.receive_json()
    assert out["name"] == "Hum" and out["say"] == "I'm thirsty."


def test_silence_produces_nothing_and_a_later_sound_still_works(streaming):
    c, _ = streaming
    with c.websocket_connect("/api/listen") as ws:
        for _ in range(5):
            ws.send_bytes(_pcm(QUIET))
        ws.send_bytes(_pcm(_sine(142), QUIET))
        assert ws.receive_json()["name"] == "Hum"


def test_small_chunks_work_like_a_real_mic(streaming):
    c, _ = streaming
    data = _pcm(QUIET, _sine(142), QUIET)
    with c.websocket_connect("/api/listen") as ws:
        send(ws, data, 1024)
        assert ws.receive_json()["name"] == "Hum"


def test_oversized_message_closes_the_stream(streaming):
    c, app_module = streaming
    from starlette.websockets import WebSocketDisconnect

    with c.websocket_connect("/api/listen") as ws:
        ws.send_bytes(b"\0\0" * (app_module.MAX_STREAM_CHUNK // 2 + 1))
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 1009


def test_bad_profile_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("TINK_DATA", str(tmp_path))
    monkeypatch.setenv("TINK_PUBLIC", "1")
    from starlette.websockets import WebSocketDisconnect

    from tink import app as app_module
    from tink import profiles as profiles_module
    from tink import store as store_module

    importlib.reload(store_module)
    importlib.reload(profiles_module)
    importlib.reload(app_module)
    try:
        c = TestClient(app_module.app)
        with pytest.raises(WebSocketDisconnect) as e, c.websocket_connect("/api/listen?p=nope"):
            pass
        assert e.value.code == 4404
    finally:
        monkeypatch.delenv("TINK_PUBLIC")
        importlib.reload(profiles_module)
        importlib.reload(app_module)


def test_not_enough_sounds_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("TINK_DATA", str(tmp_path))
    from tink import app as app_module
    from tink import profiles as profiles_module
    from tink import store as store_module

    importlib.reload(store_module)
    importlib.reload(profiles_module)
    importlib.reload(app_module)
    c = TestClient(app_module.app)
    with c.websocket_connect("/api/listen") as ws:
        send(ws, _pcm(QUIET, _sine(142), QUIET))
        assert "Teach Tink" in ws.receive_json()["error"]


def test_idle_stream_is_closed(streaming, monkeypatch):
    c, app_module = streaming
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(app_module, "STREAM_IDLE_SECONDS", 0.05)
    with c.websocket_connect("/api/listen") as ws, pytest.raises(WebSocketDisconnect) as e:
        ws.receive_json()
    assert e.value.code == 1001


def test_public_stream_refuses_another_sites_page(tmp_path, monkeypatch):
    monkeypatch.setenv("TINK_DATA", str(tmp_path))
    monkeypatch.setenv("TINK_PUBLIC", "1")
    from starlette.websockets import WebSocketDisconnect

    from tink import app as app_module
    from tink import profiles as profiles_module
    from tink import store as store_module

    importlib.reload(store_module)
    importlib.reload(profiles_module)
    importlib.reload(app_module)
    try:
        c = TestClient(app_module.app)
        with pytest.raises(WebSocketDisconnect) as e, c.websocket_connect(
            "/api/listen?p=x", headers={"origin": "https://evil.example"}
        ):
            pass
        assert e.value.code == 4429
    finally:
        monkeypatch.delenv("TINK_PUBLIC")
        importlib.reload(profiles_module)
        importlib.reload(app_module)
