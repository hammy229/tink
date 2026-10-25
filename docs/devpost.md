# Devpost draft: Tink

Draft written for the HackTX 26 Devpost. The project was not submitted, so this is kept as a write-up of the idea. No accuracy numbers were measured.

---

## Inspiration

In Peter Pan, Tinker Bell can't talk. She makes a sound like little bells, and only Peter understands her.

People with ALS end up like that. They lose words long before they lose sounds. A person who can no longer say "I'm thirsty" can still hum, click, or sigh, and the people who know them best learn what those sounds mean. Everyone else has to guess.

Speech recognition doesn't help. It is built to find words and throws everything else away.

There is one thing that does help, and it already exists. It's called message banking. While a person can still speak, they record the phrases they will want later, in their own voice. But once they can't type or speak, those recordings are hard to reach. Someone has to find the right file and press play. We wanted a sound to open them. A hum should be enough to hear "I'm thirsty" in the person's own voice.

## What it does

Tink learns the specific sounds one person makes and says out loud what each one means. When the person has banked their voice, it says it in that voice.

A caregiver teaches it a sound: type what it sounds like ("low hum") and what it means ("I'm thirsty", in whatever language the person speaks), then record the person making it about five times. After two sounds, Tink is ready.

Then comes the part we care most about. While the person can still speak, they press "Record them saying it." The phrase appears on the screen, they read it aloud, and Tink keeps the recording. Later, when Tink recognizes the hum, it plays that recording instead of a computer voice. "Say it again" and the urgent alarm play it too. If there is no recording, Tink falls back to the computer voice.

If Tink isn't sure, it doesn't pretend to be. Below a threshold it says "I didn't catch that." When two meanings are close, it asks. If the person has taught Tink a "yes" sound and a "no" sound, Tink asks about one meaning at a time ("Do you mean 'I'm thirsty'?"), and the person answers with their own sounds. A misunderstanding gets fixed in two sounds instead of a guessing game.

Some sounds can't wait. A sound marked urgent skips every question and fires even when Tink is less sure, because a missed alarm costs more than a false one. The screen turns red and Tink repeats the phrase until someone presses "I'm here."

Real rooms are noisy, so you can record a few seconds of the room itself. Tink treats that as one more sound, and when a clip matches the room better than anything the person taught, it stays quiet instead of guessing.

It works both ways. The caregiver can type a reply, like "I'm coming, two minutes," and Tink speaks it aloud to the person.

Everything runs on the laptop. Nothing is sent to any AI service.

## How we built it

The core is voc2vec (Koudounas et al., ICASSP 2025), a model pretrained on 125 hours of non-verbal human sounds: laughs, sighs, cries, vocalizations from minimally speaking people. The released checkpoints have no classifier on top, so we don't train anything. We turn each recording into one vector by averaging a hidden layer over time, average a sound's takes into one vector, and match new sounds by cosine similarity after subtracting the mean of all takes. That last step matters: without it, every sound looks about 0.9 similar to every other.

Five recordings isn't much, so each take also becomes four variants: shifted about a semitone up and down, and with two levels of background noise. Then Tink tests itself after every recording. It holds out each take, along with its own variants so it can't recognise itself, and matches it against the rest. That self-test fits two numbers to this one person, how confident a match should sound and where "I didn't catch that" begins, and the app shows the result: right on N of M takes.

Similarity scores aren't probabilities, so we put a small Bayesian model on top. The likelihood is a softmax over the scores, with an extra "unknown" option sitting at the threshold. The prior is how often each sound has been confirmed at this time of day, which starts uniform and learns every time someone presses "Right." Optional face cues from the webcam (nod, shake, smile, frown, eyes closed), computed in the browser with MediaPipe, multiply in as extra evidence.

When the posterior picks a sound, the browser plays the person's banked recording, served from the laptop at `/api/voice/{name}`. If there isn't one, it speaks the taught phrase with the browser's own voice. Audio and video never leave the machine.

The backend is FastAPI. The frontend is plain HTML, CSS, and JavaScript with no build step. The browser records until the sound ends on its own, encodes a WAV, and sends it. Each person gets their own profile, and with one setting a hosted copy gives each a private link. There is a Docker image, and 23 tests.

## Challenges we ran into

Raw voc2vec vectors are all very similar to each other, so plain cosine similarity said every clip matched every sound. Subtracting the mean of all takes fixed that, but it only works with at least two sounds, so Tink now refuses to match until it has two.

"Keep listening" mode heard Tink's own voice and answered itself. Tink now waits for its speech to finish before it listens again.

Two sounds named "Hum" and "hum!" were saved into the same folder and overwrote each other. Every sound now gets its own folder.

## Accomplishments that we're proud of

It works with five recordings per sound and no training run.

A sound now opens a message in the person's own voice. That is the part we would want if it were us.

It is honest about uncertainty. The screen shows what Tink heard, how sure it is, and how sure it would have been from the sound alone, before time of day and face cues.

## What we learned

A model trained on the right data beats a bigger model trained on the wrong data. The paper reported this on benchmarks; `risk_test.py --compare` checks it on your own recordings.

Saying "I'm not sure" is a feature. Most of our design work went into the cases where Tink is wrong.

The best feature was not a model. It was a recording of a person's own voice, played at the right moment.

## What's next for Tink

Test it with people who have ALS, and their speech therapists, before claiming anything about accuracy for them. We have not done that yet.

Run it on a phone or a small device worn on a lanyard, so it goes where the person goes.

Let the caregiver answer by voice instead of typing.

## Built with

python, fastapi, pytorch, hugging-face-transformers, voc2vec, mediapipe, javascript, web-audio-api, web-speech-api, docker
