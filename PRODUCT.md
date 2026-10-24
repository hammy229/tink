# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two groups who share one problem, being understood, and one fact: sounds belong to no language.

- **People who can't speak in words:** ALS and other conditions that take intelligible speech (dysarthria), stroke, minimally speaking autistic people. They can still make sounds: hums, clicks, sighs, short vowels.
- **People who don't share a language with the people around them:** English-as-a-second-language speakers, for example an immigrant grandparent with a home aide, or a patient with hospital staff. They teach Tink sounds with phrases in their own language, and Tink says them in English.

Caregivers (family, nurses, aides) set Tink up, teach it sounds, and confirm when it got something right. *Assumption, not confirmed by the team:* the caregiver operates the screen while the person watches what Tink says and hears it spoken. The pitch leads with one story.

## Product Purpose

Tink learns the specific sounds one person makes and says out loud what each one means, in the listener's language. Success is the person being understood in the moment, without spelling on a board, a translator app, or someone guessing.

## Positioning

Tink is built on voc2vec (Koudounas et al., ICASSP 2025), a model pretrained only on non-verbal vocalizations, so it is tuned to exactly the sounds speech recognizers throw away, and it works the same in any language. It needs about five recordings per sound, no training run, and it shows its uncertainty instead of guessing: a Bayesian ranking with an explicit "didn't catch it" option.

## Operating Context

- Runs locally on a laptop (FastAPI + browser). Mic audio goes only to the local server; face cues are computed in the browser; only text (meanings, probabilities, time, cue names, target language) goes to the LLM.
- Built at HackTX 26 (theme: fairy tales). The name and identity come from Tinker Bell, who speaks only in chimes that Peter understands.
- Judged mainly in a projected pitch: the screen is read from across a room.

## Capabilities and Constraints

- Teach a sound (name, phrase in any language, optional face cues, about 5 takes); listen; Bayesian ranking over taught sounds plus "unknown"; time-of-day prior that learns from "Right"; webcam cues (nod, shake, smile, frown, eyes closed); Grok wording and translation into the chosen spoken language, with taught-phrase fallback; browser speech synthesis output.
- Plain HTML/CSS/JS frontend, no build step. Element ids in `web/index.html` are the contract with `web/app.js`.
- Accuracy on real users is not yet measured. Do not claim accuracy numbers until `risk_test.py` has run on real recordings.

## Brand Commitments

- Name: Tink. Peter Pan / Tinker Bell identity: night sky, Tinker Bell's gold light, a bell chime when Tink understands.
- *Assumption, not confirmed:* the theme is clear but grown-up; it should read as a real assistive product, not a children's game.

## Evidence on Hand

- The voc2vec paper and its released checkpoints (Apache-2.0).
- No user testimonials, clinical results, or accuracy benchmarks yet. Do not invent them.

## Product Principles

1. Being understood beats being clever: say the taught phrase rather than an embellished one when in doubt.
2. Show uncertainty honestly; ask instead of guessing.
3. Everything private stays on the device; show exactly what leaves it.
4. Every interaction must work for someone with limited movement, limited energy, or limited English.

## Accessibility & Inclusion

Some users have progressive motor impairment and weakening facial expression, so face cues are optional evidence, never required. Some users read little English, so the person-facing text is short and the spoken output carries the meaning. Large targets, keyboard and switch-style operation (space to listen), WCAG AA contrast in both themes, reduced motion honored, no time-limited controls.
