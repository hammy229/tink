"""Turn match scores into probabilities, then update them with time of day and webcam cues.

    P(sound | clip, cues, time)  ∝  P(clip | sound) · P(cues | sound) · P(sound | time)

- P(clip | sound): softmax of centered cosine scores at temperature TEMP. An extra "unknown"
  option sits at the match threshold, so weak matches lose to "I didn't catch that".
- P(sound | time): Dirichlet-smoothed counts of confirmed uses in this part of the day.
  Starts uniform and learns every time someone presses "Right".
- P(cues | sound): each cue seen on camera counts P_TAGGED if the sound was taught with that
  cue, else P_OTHER. Cues not seen are ignored (the camera misses things).
"""
import math
import os

CUES = ("nod", "shake", "smile", "frown", "eyes_closed")
UNKNOWN = "unknown"
TEMP = float(os.environ.get("TINK_TEMP", "0.1"))
ALPHA = 1.0
P_TAGGED, P_OTHER = 0.6, 0.2


def part_of_day(hour: int) -> str:
    return "night" if hour < 6 else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"


def _normalize(logs: dict[str, float]) -> dict[str, float]:
    top = max(logs.values())
    exp = {k: math.exp(v - top) for k, v in logs.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


def posterior(scores, threshold, usage, sound_cues, cues_seen, hour, temp: float = TEMP) -> dict:
    names = list(scores)
    audio = {n: scores[n] / temp for n in names} | {UNKNOWN: threshold / temp}

    bucket = part_of_day(hour)
    counts = {n: usage.get(n, {}).get(bucket, 0) for n in names} | {UNKNOWN: 0}
    total = sum(counts.values()) + ALPHA * len(counts)
    prior = {k: (c + ALPHA) / total for k, c in counts.items()}

    seen = [c for c in cues_seen if c in CUES]
    cue = {n: sum(math.log(P_TAGGED if c in sound_cues.get(n, []) else P_OTHER) for c in seen) for n in names}
    cue[UNKNOWN] = len(seen) * math.log(P_OTHER)

    final = _normalize({k: audio[k] + math.log(prior[k]) + cue[k] for k in audio})
    audio_only = _normalize(audio)
    ranked = sorted(final, key=final.get, reverse=True)
    return {
        "part_of_day": bucket,
        "cues": seen,
        "ranked": [
            {"name": k, "p": round(final[k], 3), "audio_p": round(audio_only[k], 3), "prior": round(prior[k], 3)}
            for k in ranked
        ],
    }
