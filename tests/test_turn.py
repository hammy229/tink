"""The safety rules of a turn, as plain inputs and outputs. No model, no server."""
from datetime import datetime, timedelta

from tink import turn
from tink.bayes import UNKNOWN
from tink.turn import Pending, Sounds, decide_without_wording, finish_turn

NOW = datetime(2026, 11, 5, 14, 0)
SOUNDS = Sounds(
    phrase={"Hum": "I'm thirsty.", "Hiss": "I'm cold.", "Ah": "Yes", "Mm": "No", "Room": "(room)", "Cry": "It hurts!"},
    roles={"Ah": "yes", "Mm": "no", "Room": "background"},
    urgent=frozenset({"Cry"}),
)


def ranked(*pairs):
    return [{"name": n, "p": p, "audio_p": p, "prior": 0.1} for n, p in pairs]


def test_room_noise_stays_quiet_and_keeps_the_question_open():
    open_q = Pending(("Hum",), 0, NOW)
    out = decide_without_wording(ranked(("Room", 0.9), ("Hum", 0.1)), SOUNDS, open_q, NOW)
    assert out.fields["background"] is True and out.fields["say"] == ""
    assert out.pending is open_q


def test_urgent_fires_below_the_ask_threshold_even_when_not_on_top():
    out = decide_without_wording(ranked(("Hum", 0.5), ("Cry", 0.45)), SOUNDS, None, NOW)
    assert out.fields["urgent"] is True and out.fields["say"] == "It hurts!" and out.fields["ask"] is False


def test_urgent_needs_some_evidence():
    assert decide_without_wording(ranked(("Hum", 0.7), ("Cry", 0.2)), SOUNDS, None, NOW) is None


def test_urgent_closes_an_open_question():
    out = decide_without_wording(ranked(("Cry", 0.9),), SOUNDS, Pending(("Hum",), 0, NOW), NOW)
    assert out.pending is None


def test_yes_says_what_was_asked_not_what_was_heard():
    out = decide_without_wording(ranked(("Ah", 0.9),), SOUNDS, Pending(("Hum", "Hiss"), 0, NOW), NOW)
    assert out.fields["name"] == "Hum" and out.fields["say"] == "I'm thirsty." and out.said == "I'm thirsty."
    assert out.pending is None


def test_no_moves_to_the_next_option_then_gives_up():
    first = Pending(("Hum", "Hiss"), 0, NOW - timedelta(seconds=5))
    out = decide_without_wording(ranked(("Mm", 0.9),), SOUNDS, first, NOW)
    assert out.fields["dialog"] == {"answered": "no", "asking": "Hiss"}
    assert out.pending.i == 1 and out.pending.at == NOW
    assert first.i == 0  # the previous state wasn't mutated

    last = decide_without_wording(ranked(("Mm", 0.9),), SOUNDS, out.pending, NOW)
    assert last.fields["say"] == turn.GIVE_UP and last.pending is None


def test_a_weak_yes_does_not_answer():
    assert decide_without_wording(ranked(("Ah", 0.5), ("Hum", 0.3)), SOUNDS, Pending(("Hum",), 0, NOW), NOW) is None


def test_an_expired_question_cannot_be_answered():
    stale = Pending(("Hum",), 0, NOW - timedelta(seconds=turn.DIALOG_SECONDS + 1))
    assert decide_without_wording(ranked(("Ah", 0.9),), SOUNDS, stale, NOW) is None


def test_sure_match_is_spoken_and_remembered():
    out = finish_turn(ranked(("Hum", 0.9), ("Hiss", 0.1)), SOUNDS, None, NOW)
    assert out.fields["say"] == "I'm thirsty." and out.fields["ask"] is False and out.said == "I'm thirsty."


def test_llm_cannot_turn_an_unsure_guess_into_a_statement():
    no_yes_no = Sounds({"Hum": "a", "Hiss": "b"}, {}, frozenset())
    out = finish_turn(ranked(("Hum", 0.5), ("Hiss", 0.4)), no_yes_no, {"say": "I am thirsty!", "ask": False}, NOW)
    assert out.fields["ask"] is True and out.said is None


def test_unsure_guess_opens_a_question_over_real_sounds_only():
    out = finish_turn(ranked(("Hum", 0.4), ("Ah", 0.3), ("Hiss", 0.2), (UNKNOWN, 0.1)), SOUNDS, None, NOW)
    assert out.pending.options == ("Hum", "Hiss")  # never asks "do you mean yes?"
    assert out.fields["source"] == "dialog" and "I'm thirsty." in out.fields["say"]


def test_unknown_without_yes_no_asks_to_try_again():
    no_yes_no = Sounds({"Hum": "a", "Hiss": "b"}, {}, frozenset())
    out = finish_turn(ranked((UNKNOWN, 0.8), ("Hum", 0.2)), no_yes_no, None, NOW)
    assert out.fields["say"] == turn.NOT_CAUGHT and out.fields["name"] is None
