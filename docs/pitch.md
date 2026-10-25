# Pitch and demo script

Written for HackTX 26 judging. The project was not submitted; kept for reference.

Judging: Sun 2:00 to 5:00 in WEL (table). Pitching: Sun 5:30 in GDC 2.216 (projected). Press **P** for present mode before you start.

## Before judges arrive (every time)

1. Server running on the demo laptop, not a teammate's: `uv run uvicorn tink.app:app`
2. Open http://localhost:8000 in Chrome. Allow the microphone (and camera, if using cues).
3. The demo person's sounds are already taught: at least 3 sounds, 5 takes each, recorded **in the judging room's noise**, not the quiet hacker room. Suggested set:
   - Low hum: "I'm thirsty."
   - Two tongue clicks: "Please call my daughter."
   - Long sigh: "I'm tired. Can we rest?"
   - A short "ah" marked **It means yes**, and a short "mm" marked **It means no**.
4. **Bank their voice** for the demo sounds: press **Record them saying it** for each one and read the phrase aloud. Play each back once to check it is audible.
5. Record **room noise** 3 times in the judging room itself, and teach one **urgent** sound (e.g. a sharp "ah!" meaning "Help, it hurts!"), and bank its voice too.
6. Do 3 test runs. If one misses, record 2 more takes of that sound and try again.
7. Volume up, Mute voice off. Close other tabs that might play sound. Tink needs no internet except to load the webcam cue model from a CDN; leave the camera off if wifi is bad.

## 2-minute pitch

**(0:00) The hook.** In Peter Pan, Tinker Bell can't talk. She makes a sound like little bells, and only Peter understands her.

**(0:10) The problem.** People with ALS are Tinker Bell. They lose words long before they lose sounds. They can still hum or click or sigh, and the people closest to them learn what it means. Everyone else guesses. Speech recognition doesn't help: it's built to find words and throws the rest away. Message banking, where people record their own voice before they lose it, is real and it helps. But once you can't type or speak, those recordings are hard to reach. We let a sound open them.

**(0:35) The demo.** (keep Keep listening on, so nobody touches the laptop) *(Demo person makes the hum.)* Tink heard a hum. It's 97% sure. And it answers in her own recorded voice: "I'm thirsty." That recording was made while she could still speak. The hum is what she can still do. *(Point at the ranking.)* You can see what else it considered and how sure it was from the sound alone.

*(Make an in-between sound on purpose.)* When it isn't sure, it doesn't guess. It asks: "Do you mean 'I'm thirsty'?" *(Make the "no" sound.)* "Do you mean 'I'm cold'?" *(Make the "yes" sound.)* "I'm cold." Two sounds and the misunderstanding is fixed. That's the most important thing it does.

*(Make the urgent sound.)* Some sounds can't wait. This one skips every question, turns the screen red, and repeats until someone answers. *(Press I'm here.)*

*(Optional, before pressing P: type "I'm coming, two minutes" in Answer back.)* The caregiver can answer too, and Tink speaks it aloud.

**(1:15) How.** Underneath is voc2vec, a model from ICASSP this year trained only on non-verbal sounds. Five recordings per sound, no training. A small Bayesian model turns similarities into honest probabilities, and learns from time of day and optional face cues. It all runs on this laptop. No AI service sees anything.

**(1:45) Close.** Peter understood Tinker Bell because he knew her. Tink keeps a person's own voice reachable after the words are gone.

## Questions judges will ask

- **How accurate is it?** Give the measured leave-one-out number and say how many clips. Don't round up. Say plainly that we haven't tested with ALS patients yet and wouldn't claim clinical accuracy without that.
- **Why not just use speech-to-text?** It's trained to find words. These sounds aren't words. voc2vec's paper shows general speech models do worse on vocalizations; our `--compare` test checks that on our voices.
- **How do you know the percentages mean anything?** Tink tests itself after every recording: each take is held out and matched against the rest, which fits its confidence to this person. The Sounds panel shows the result, including how often it was right when it said 90%+.
- **Why Bayesian?** Similarity scores aren't probabilities and can't say "none of these." The posterior can, and it gets better with each "Right."
- **What about privacy?** Everything runs on the laptop. Audio, including banked voices, stays there. Face cues are computed in the browser and only words like "smile" are kept. Nothing is sent to any AI service.
- **Why message banking?** It is a real practice in ALS care: people record their own voice while they can still speak. The gap is access. Once someone can't type or speak, a bank of recordings is hard to use. Tink connects the banked message to the sound they can still make.
- **What happens when the voice is gone completely?** The banked recordings stay. The sounds they can still make keep working, and face cues can help. If they lose those sounds too, Tink needs new ones, so the earlier the better. We have not tested with ALS patients yet, and we would not claim it holds up through the whole disease.
- **What would you build next?** Testing with people with ALS and their speech therapists, spoken caregiver replies instead of typed ones, and a wearable version.

## If something breaks during judging

| Problem | Do this |
|---|---|
| Wrong sound recognized | Press Wrong, say "that's why it asks when it's unsure", record 2 more takes of that sound live. Judges like seeing it learn. |
| "I didn't hear anything" | Move the laptop closer to the person. The mic needs a clear sound above room noise. |
| No voice | Check Mute voice isn't on, check volume, press Say it again. |
| Page says it can't reach the server | Restart `uv run uvicorn tink.app:app` and reload. Takes and sounds are saved on disk. |
| Banked voice doesn't play | Check Mute voice isn't on and the volume is up. Tink falls back to the computer voice, so the demo still works. Re-record it with Record them saying it. |
