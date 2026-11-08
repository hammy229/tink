"""Hands-free segmentation, and the guard that keeps the Python and browser engines in step."""
import json

import numpy as np

from scripts.gen_parity import OUT, build
from tink.vad import FRAME, SegmentDetector


def frames(level: float, n: int) -> np.ndarray:
    signs = np.tile(np.repeat([1.0, -1.0], FRAME), n // 2 + 1)[: n * FRAME]
    return (signs * level).astype(np.float32)


def segments(*parts: np.ndarray) -> list[np.ndarray]:
    det = SegmentDetector()
    return det.push(np.concatenate(parts)) + det.flush()


def assert_close(got: object, want: object, path: str = "") -> None:
    """Equal structure and strings, numbers within 1e-5: BLAS on Linux and macOS can differ in the 6th decimal."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys(), path
        for k in want:
            assert_close(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_close(g, w, f"{path}[{i}]")
    elif isinstance(want, float) or isinstance(got, float):
        assert abs(float(got) - float(want)) <= 1e-5, f"{path}: {got} != {want}"  # type: ignore[arg-type]
    else:
        assert got == want, path


def test_fixture_is_current():
    want = json.loads(json.dumps(build()))
    try:
        assert_close(json.loads(OUT.read_text()), want)
    except AssertionError as e:
        raise AssertionError(f"{e}; run: uv run python -m scripts.gen_parity") from e


def test_finds_one_sound_and_keeps_its_onset():
    out = segments(frames(0.001, 50), frames(0.2, 40), frames(0.001, 50))
    assert len(out) == 1
    assert np.abs(out[0][: 5 * FRAME]).max() < 0.01  # includes the pre-roll from before the sound began


def test_a_short_tick_is_not_a_sound():
    assert segments(frames(0.001, 50), frames(0.2, 3), frames(0.001, 50)) == []


def test_two_clicks_close_together_stay_one_sound():
    out = segments(frames(0.001, 50), frames(0.2, 6), frames(0.001, 10), frames(0.2, 6), frames(0.001, 50))
    assert len(out) == 1


def test_sounds_far_apart_are_separate():
    out = segments(frames(0.001, 50), frames(0.2, 10), frames(0.001, 50), frames(0.2, 10), frames(0.001, 50))
    assert len(out) == 2


def test_noise_floor_adapts_so_a_steady_fan_goes_quiet():
    det = SegmentDetector()
    det.push(frames(0.001, 50))
    first = det.push(frames(0.02, 500)) + det.flush()
    assert len(first) <= 3  # the switch-on may fire, then the floor learns the fan
    assert det.floor is not None and det.floor > 0.01
    assert det.push(frames(0.02, 100)) == []
    assert len(det.push(frames(0.3, 20)) + det.push(frames(0.02, 50))) == 1  # and a real sound still gets through


def test_a_sound_that_never_ends_is_cut_off():
    out = segments(frames(0.001, 50), frames(0.2, 1000))
    assert out and all(len(s) <= 200 * FRAME for s in out)


def test_chunk_size_does_not_change_the_result():
    stream = np.concatenate([frames(0.001, 50), frames(0.2, 40), frames(0.001, 50)])
    whole = SegmentDetector().push(stream)
    det = SegmentDetector()
    parts = [s for chunk in np.array_split(stream, 173) for s in det.push(chunk)]
    assert len(whole) == len(parts) and np.array_equal(whole[0], parts[0])
