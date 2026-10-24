---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: ["web/style.css","web/app.js"]
---

# Surface brief: Tink app (web/index.html)

Mode: Operate. Single-screen tool, also projected during the pitch (present mode, key P).
Audience: a caregiver operating; the person who can't be understood (nonspeaking or ESL) watching and listening; judges reading from across a room.
Task: teach sounds, listen, see and hear what a sound means, correct Tink when it is wrong or unsure.
Constraints: plain HTML/CSS/JS, element ids are the contract with app.js, WCAG AA, reduced motion, 48px targets, no time-limited controls.
Memorable moment: the screen lights in the heard sound's own colour and the phrase appears on it in black silhouette type.

## Direction contract

THESIS: Tink is a backlit paper screen. The person's sound is the light behind it; what the sound means is the shadow it casts. Refuses the category default of a dark dashboard with a glowing accent orb and a result card.

OWN-WORLD: warm theatre-black house (#15110e) with hairline frames; one backlit scrim of warm paper (#f7ead0) carrying all emphasis; silhouette ink (#15110e) for the phrase and a cut-paper London skyline along the scrim's foot; six film tints (amber, teal, rose, violet, leaf, sky), one owned by each taught sound in its list swatch, ranking bar, log stamp, and the scrim itself when that sound is heard. Gloock for the phrase and wordmark, Atkinson Hyperlegible for everything operated.

STORY: the visitor sees the screen breathe with the voice, sees it light in a sound's colour with the meaning in silhouette, hears it spoken, and believes Tink is honest because the ranking, the stamped log, and the exact LLM packet sit beside every answer.

FIRST VIEWPORT: top bar (wordmark left; Speak in select, model chip, Present right). Left two thirds: the scrim, about 560px tall, phrase centred in silhouette at 44 to 64px, ranking under it, Listen (64px) with Say again and Mute below the scrim edge. Right third: Sounds Tink knows with tint swatches, then Teach a sound. Log and packet panels under the scrim. Signature move: the live light behind the scrim, drawn every frame from the mic signal, blooming into the heard sound's tint.

FORM: Silhouette Theatre (Lotte Reiniger cut-paper film), position 3 on the grounded list; raises: one tint per sound, stamped log, dim-outside-while-speaking, result beside proof, live-drawn light, hairline structure. Seed key 991ae83a.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
