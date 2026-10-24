# Devpost draft: Tink

Paste each section into its Devpost field. Before submitting, replace every `[[...]]` with a real number from `risk_test.py`, or delete the sentence. Do not submit a number we did not measure.

---

## Inspiration

In Peter Pan, Tinker Bell can't talk. She makes a sound like little bells, and only Peter understands her.

A lot of people live like that. Someone with ALS loses the ability to form words long before they lose the ability to make sounds. A grandmother who only speaks Gujarati can't tell her English-speaking aide that she's cold. Both of them can still hum, click, or sigh, and the people who know them best learn what those sounds mean. Everyone else has to guess.

Speech recognition doesn't help, because it is built to find words and throws everything else away. We wanted the opposite: something that listens to the sounds that aren't words.

## What it does

Tink learns the specific sounds one person makes and says out loud what each one means.

A caregiver teaches it a sound: type what it sounds like ("low hum") and what it means ("I'm thirsty", in any language), then record the person making it about five times. After two sounds, Tink is ready. When the person makes a sound, Tink chimes, shows the phrase in large type, and speaks it.

If it isn't sure, it doesn't pretend to be. Below a threshold it says "I didn't catch that." When two meanings are close, it asks: "Do you want water, or do you want to rest?"

Tink can speak in a different language from the one the phrase was taught in, so a sound taught as "Tengo sed" can come out as "I'm thirsty."

## How we built it

The core is voc2vec (Koudounas et al., ICASSP 2025), a model pretrained on 125 hours of non-verbal human sounds: laughs, sighs, cries, vocalizations from minimally speaking people. The released checkpoints have no classifier on top, so we don't train anything. We turn each recording into one vector by averaging a hidden layer over time, average a sound's takes into one vector, and match new sounds by cosine similarity after subtracting the mean of all takes. That last step matters: without it, every sound looks about 0.9 similar to every other.

Similarity scores aren't probabilities, so we put a small Bayesian model on top. The likelihood is a softmax over the scores, with an extra "unknown" option sitting at the threshold. The prior is how often each sound has been confirmed at this time of day, which starts uniform and learns every time someone presses "Right." Optional face cues from the webcam (nod, shake, smile, frown, eyes closed), computed in the browser with MediaPipe, multiply in as extra evidence.

The top meanings, their probabilities, the time, and the cue names go to Grok as a small JSON packet. Grok turns that into one natural sentence in the listener's language, or a question when Tink is unsure. Audio and video never leave the laptop, and the screen shows the exact packet that does. If Grok is unreachable, Tink says the taught phrase, so the demo can't stall on the network.

The backend is FastAPI. The frontend is plain HTML, CSS, and JavaScript with no build step. The browser records until the sound ends on its own, encodes a WAV, and sends it.

## Challenges we ran into

Raw voc2vec vectors are all very similar to each other, so plain cosine similarity said every clip matched every sound. Subtracting the mean of all takes fixed that, but it only works with at least two sounds, so Tink now refuses to match until it has two.

"Keep listening" mode heard Tink's own voice and answered itself. Tink now waits for its speech to finish before it listens again.

Two sounds named "Hum" and "hum!" were saved into the same folder and overwrote each other. Every sound now gets its own folder.

## Accomplishments that we're proud of

It works with five recordings per sound and no training run.

It is honest about uncertainty. The screen shows what Tink heard, how sure it is, and how sure it would have been from the sound alone, before time of day and face cues.

On recordings from our own team, voc2vec matched sounds correctly [[X]]% of the time in leave-one-out testing, against [[Y]]% for wav2vec2 and [[Z]]% for HuBERT on the same clips. (`uv run python risk_test.py --compare`)

## What we learned

A model trained on the right data beats a bigger model trained on the wrong data. The paper reported this on benchmarks; we checked it on our own voices.

Saying "I'm not sure" is a feature. Most of our design work went into the cases where Tink is wrong.

## What's next for Tink

Test it with people who actually communicate this way, and their speech therapists, before claiming anything about accuracy for them.

Run it on a phone or a small device worn on a lanyard, so it goes where the person goes.

Let the person answer Tink's questions with their own yes and no sounds, so a misunderstanding gets fixed in two sounds instead of a guessing game.

## Built with

python, fastapi, pytorch, hugging-face-transformers, voc2vec, mediapipe, grok, xai, javascript, web-audio-api, web-speech-api
