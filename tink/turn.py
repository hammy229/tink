"""One turn of conversation: given what the model heard, decide what Tink does.

Pure on purpose. No I/O, no clock, no model, so every safety rule below is a fast unit test:

- The room itself never speaks and never disturbs an open question.
- Urgent sounds skip the questions and alert at lower confidence, because a missed alarm costs more than a false one.
- An unsure guess is never spoken as a statement. It becomes a question, even if an LLM worded it as one.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from .bayes import UNKNOWN

ASK_BELOW = 0.6  # top probability under this means Tink asks instead of saying
DIALOG_SECONDS = 45  # how long a yes/no question stays open
ANSWER_MIN = 0.6  # a yes or no sound must be at least this sure to answer a question
URGENT_MIN = 0.4  # urgent sounds alert at lower confidence
NOT_CAUGHT = "I didn't catch that. Can you make the sound again?"
GIVE_UP = "Okay. Make the sound again when you're ready."


@dataclass(frozen=True)
class Sounds:
    """What the person has taught, in the form the decision needs."""

    phrase: dict[str, str]
    roles: dict[str, str]
    urgent: frozenset[str]

    @property
    def has_yes_no(self) -> bool:
        return {"yes", "no"} <= set(self.roles.values())


@dataclass(frozen=True)
class Pending:
    """An open yes/no question: the options best first, and which one was just asked."""

    options: tuple[str, ...]
    i: int
    at: datetime

    @property
    def asked(self) -> str:
        return self.options[self.i]


@dataclass(frozen=True)
class Outcome:
    fields: dict[str, Any]  # goes into the response
    pending: Pending | None  # the question left open after this turn
    said: str | None = None  # remembered as something the person said


def decide_without_wording(
    ranked: list[dict], sounds: Sounds, pending: Pending | None, now: datetime
) -> Outcome | None:
    """The cases that don't need any wording: the room, an alarm, or an answer to an open question.

    Returns None when this is an ordinary turn, which goes on to finish_turn().
    """
    top = ranked[0]
    phrase = sounds.phrase
    matched = top["name"] != UNKNOWN and top["name"] in phrase
    role = sounds.roles.get(top["name"]) if matched else None

    if role == "background":
        return Outcome(
            {"matched": False, "name": None, "say": "", "ask": False, "source": "background",
             "background": True, "taught": None, "dialog": None},
            pending,
        )

    urgent = next((r for r in ranked if r["name"] in sounds.urgent and r["p"] >= URGENT_MIN), None)
    if urgent and urgent["name"] in phrase:
        say = phrase[urgent["name"]]
        return Outcome(
            {"p": urgent["p"], "matched": True, "name": urgent["name"], "say": say, "ask": False, "source": "urgent",
             "urgent": True, "taught": say, "dialog": None},
            None,
            say,
        )

    answering = role if role and top["p"] >= ANSWER_MIN else None
    if pending and answering and (now - pending.at).total_seconds() < DIALOG_SECONDS:
        if answering == "yes" and pending.asked in phrase:
            say = phrase[pending.asked]
            return Outcome(
                {"matched": True, "name": pending.asked, "say": say, "ask": False, "source": "dialog", "taught": say,
                 "dialog": {"answered": "yes", "meant": pending.asked}},
                None,
                say,
            )
        following = pending.i + 1
        nxt = pending.options[following] if following < len(pending.options) else None
        if nxt in phrase:
            return Outcome(
                {"matched": False, "name": None, "say": f'Do you mean "{phrase[nxt]}"?', "ask": True,
                 "source": "dialog", "dialog": {"answered": "no", "asking": nxt}},
                replace(pending, i=following, at=now),
            )
        return Outcome(
            {"matched": False, "name": None, "say": GIVE_UP, "ask": True, "source": "dialog",
             "dialog": {"answered": "no"}},
            None,
        )
    return None


def fallback(ranked: list[dict], phrase: dict[str, str]) -> dict:
    """What Tink says with no LLM: the phrase, an either/or question, or a request to try again."""
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    if top["name"] == UNKNOWN or top["name"] not in phrase:
        return {"say": NOT_CAUGHT, "ask": True}
    if top["p"] < ASK_BELOW and second and second["name"] in phrase:
        return {"say": f'Do you mean "{phrase[top["name"]]}" or "{phrase[second["name"]]}"?', "ask": True}
    return {"say": phrase[top["name"]], "ask": False}


def finish_turn(ranked: list[dict], sounds: Sounds, worded: dict | None, now: datetime) -> Outcome:
    """An ordinary turn. `worded` is the optional LLM's reply; None means use the taught phrase."""
    top, phrase = ranked[0], sounds.phrase
    matched = top["name"] != UNKNOWN and top["name"] in phrase
    source = "grok" if worded else "phrase"
    out = dict(worded or fallback(ranked, phrase))
    if not matched or top["p"] < ASK_BELOW:
        out["ask"] = True  # never let wording turn an unsure guess into a statement

    dialog = None
    pending = None
    if out["ask"] and sounds.has_yes_no:
        options = tuple(r["name"] for r in ranked if r["name"] in phrase and r["name"] not in sounds.roles)[:3]
        if options:
            pending = Pending(options, 0, now)
            out = {"say": f'Do you mean "{phrase[options[0]]}"?', "ask": True}
            source, dialog = "dialog", {"asking": options[0]}
    return Outcome(
        {"matched": matched, "name": top["name"] if matched else None, "say": out["say"], "ask": out["ask"],
         "source": source, "dialog": dialog},
        pending,
        None if out["ask"] else out["say"],
    )
