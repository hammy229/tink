# Tink for ALS

Most people with ALS lose their speech over time. Long after words go, many can still make a few sounds: a hum, a click, a sigh. Tink learns what those sounds mean for one person and says it out loud, in that person's own voice.

## How it works

1. **Teach a sound.** The caregiver records about 5 takes of a sound the person can make reliably and types what it means ("I'm in pain"). Each take also gets 4 variants (pitch up and down, two noise levels), so Tink still recognizes the sound when it drifts.
2. **Bank their voice.** While the person can still speak, they press **Record them saying it** and read the phrase aloud. Tink saves that recording next to the sound.
3. **Listen.** Tink turns each new sound into a voc2vec embedding, compares it to the taught sounds, and computes a probability for each one, with "didn't catch it" as an option so it doesn't guess.
4. **Speak.** When a sound matches, Tink plays the person's own recording. If they never banked that phrase, or the caregiver picked another language, it uses the computer voice.

Other features that matter for ALS:

- **Urgent sounds** (e.g. "I can't breathe") set off a red screen and an alarm. They fire at a lower confidence than other sounds, because a missed alarm costs more than a false one.
- **Yes/no sounds.** When two meanings are close, Tink asks, and the person answers with their own yes and no sounds.
- **Answer-back.** The caregiver types a reply and Tink speaks it.
- **Privacy.** Sounds and banked voices stay on the laptop.

## Where it is in the code

| What | Where |
|---|---|
| Save and find a banked voice | `tink/store.py`, `set_voice` and `voice_path` |
| API to bank and play a voice | `tink/app.py`, `POST /api/voice` and `GET /api/voice/{name}` |
| Play their voice, else the computer voice | `web/app.js`, `playVoice` and `sayFor` (the "Their own voice" section) |
| "Record them saying it" button | `web/app.js`, `bankVoice` |
| Urgent alarm | `tink/app.py` (`URGENT_MIN`, urgent check in listen), `web/app.js` ("Urgent alarm" section) |
| Augmentation for drifting sounds | `tink/audio.py`, `tink/store.py` (`calibrate`) |
| Matching and probabilities | `tink/embed.py`, `tink/bayes.py` |

## Try it

Run the app (see the [README](../README.md#run-it)). Teach a sound, press the microphone button next to it to bank the phrase, then make the sound. Tink should play your recording.
