# Pitch and demo script

Judging: Sun 2:00 to 5:00 in WEL (table). Pitching: Sun 5:30 in GDC 2.216 (projected). Press **P** for present mode before you start.

## Before judges arrive (every time)

1. Server running on the demo laptop, not a teammate's: `uv run uvicorn tink.app:app`
2. Open http://localhost:8000 in Chrome. Allow the microphone (and camera, if using cues).
3. The demo person's sounds are already taught: at least 3 sounds, 5 takes each, recorded **in the judging room's noise**, not the quiet hacker room. Suggested set:
   - Low hum: "I'm thirsty."
   - Two tongue clicks: "Please call my daughter."
   - Long sigh: "I'm tired. Can we rest?"
   - One taught in another language, e.g. "Tengo frío." (Spanish for "I'm cold.")
   - A short "ah" marked **It means yes**, and a short "mm" marked **It means no**.
4. Do 3 test runs. If one misses, record 2 more takes of that sound and try again.
5. Volume up. Close other tabs that might play sound.
6. `XAI_API_KEY` set and one test run shows "worded by grok" under the phrase. If the wifi is bad, unset the key and restart: Tink falls back to the taught phrases and still works offline. (The webcam cue model loads from a CDN; leave the camera off if wifi is bad.)

## 2-minute pitch

**(0:00) The hook.** In Peter Pan, Tinker Bell can't talk. She makes a sound like little bells, and only Peter understands her.

**(0:10) The problem.** Lots of people are Tinker Bell. Someone with ALS loses words long before they lose sounds. A grandmother who only speaks Gujarati can't tell her aide she's cold. They can still hum or click or sigh, and the people closest to them learn what it means. Everyone else guesses. Speech recognition doesn't help: it's built to find words and throws the rest away.

**(0:30) The demo.** (keep Keep listening on, so nobody touches the laptop) *(Demo person makes the hum.)* Tink heard a hum. It's 97% sure, and it says "I'm thirsty." *(Point at the ranking.)* You can see what else it considered and how sure it was from the sound alone.

*(Second sound: the Spanish one, with Speak in set to English.)* This one was taught in Spanish. Tink says it in English.

*(Make an in-between sound on purpose.)* When it isn't sure, it doesn't guess. It asks: "Do you mean 'I'm thirsty'?" *(Make the "no" sound.)* "Do you mean 'I'm cold'?" *(Make the "yes" sound.)* "I'm cold." Two sounds and the misunderstanding is fixed. That's the most important thing it does.

**(1:10) How.** Underneath is voc2vec, a model from ICASSP this year trained only on non-verbal sounds. Five recordings per sound, no training. A small Bayesian model turns similarities into honest probabilities, and learns from time of day and optional face cues. Grok turns that into a sentence in the listener's language. Audio and video never leave the laptop: the screen shows exactly what does.

**(1:35) The number.** On our own team's recordings, voc2vec got [[X]]% right. A general speech model on the same clips got [[Y]]%. *(Only say this if you ran `risk_test.py --compare`. Otherwise skip it.)*

**(1:45) Close.** Peter understood Tinker Bell because he knew her. Tink lets anyone understand the people they don't know yet.

## Questions judges will ask

- **How accurate is it?** Give the measured leave-one-out number and say how many clips. Don't round up. Say plainly that we haven't tested with ALS patients yet and wouldn't claim clinical accuracy without that.
- **Why not just use speech-to-text?** It's trained to find words. These sounds aren't words. voc2vec's paper shows general speech models do worse on vocalizations; our `--compare` test checks that on our voices.
- **Why Bayesian?** Similarity scores aren't probabilities and can't say "none of these." The posterior can, and it gets better with each "Right."
- **What about privacy?** Audio stays on the laptop. Face cues are computed in the browser and only words like "smile" are kept. Only text goes to Grok, and the screen shows it.
- **What if Grok is down?** Tink says the taught phrase. It works fully offline except translation.
- **What would you build next?** Caregivers typing a reply that Tink speaks in the person's language, and a wearable version.

## If something breaks during judging

| Problem | Do this |
|---|---|
| Wrong sound recognized | Press Wrong, say "that's why it asks when it's unsure", record 2 more takes of that sound live. Judges like seeing it learn. |
| "I didn't hear anything" | Move the laptop closer to the person. The mic needs a clear sound above room noise. |
| No voice | Check Mute voice isn't on, check volume, press Say it again. |
| Page says it can't reach the server | Restart `uv run uvicorn tink.app:app` and reload. Takes and sounds are saved on disk. |
| Wifi down | Unset `XAI_API_KEY`, restart. Everything except translation works offline. |
