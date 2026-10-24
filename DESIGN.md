---
name: Tink
description: A backlit paper screen in a theatre-black house; the sound is the light, the meaning is the shadow.
colors:
  house: "#15110e"
  surface: "#201a15"
  line: "#3d332b"
  control-line: "#8c7d6c"
  fg: "#f4ece0"
  muted: "#bcae9a"
  paper: "#f7ead0"
  paper-dim: "#e8d8b8"
  ink: "#15110e"
  ash: "#9c958c"
  lamp: "#f3b648"
  miss: "#f0907f"
  tint-0: "#f3b648"
  tint-1: "#3aa39b"
  tint-2: "#e46f7a"
  tint-3: "#a58ad8"
  tint-4: "#93c45e"
  tint-5: "#62acdf"
typography:
  display:
    fontFamily: "Gloock, Bodoni 72, Georgia, serif"
    fontSize: "clamp(40px, 5.2vw, 68px)"
    fontWeight: 400
    lineHeight: 1.06
    letterSpacing: "-0.012em"
  display-present:
    fontFamily: "Gloock, Bodoni 72, Georgia, serif"
    fontSize: "clamp(56px, 8vw, 140px)"
    fontWeight: 400
    lineHeight: 1.06
    letterSpacing: "-0.012em"
  headline:
    fontFamily: "Gloock, Bodoni 72, Georgia, serif"
    fontSize: "40px"
    fontWeight: 400
    lineHeight: 1
    letterSpacing: "-0.01em"
  numeral:
    fontFamily: "Gloock, Bodoni 72, Georgia, serif"
    fontSize: "30px"
    fontWeight: 400
    lineHeight: 1
  title:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0"
  body:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.55
  body-screen:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "16.5px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1.4
  caption:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "14.5px"
    fontWeight: 400
    lineHeight: 1.5
  stamp:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 700
    letterSpacing: "0.04em"
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, Menlo, monospace"
    fontSize: "13.5px"
    fontWeight: 400
    fontFeature: "tnum"
rounded:
  swatch: "4px"
  stamp: "6px"
  tight: "10px"
  field: "12px"
  panel: "16px"
  screen: "22px"
  pill: "999px"
spacing:
  hair: "6px"
  xs: "8px"
  sm: "12px"
  md: "14px"
  lg: "18px"
  xl: "22px"
components:
  button-primary:
    backgroundColor: "{colors.lamp}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "11px 22px"
    height: "48px"
  button-listen:
    backgroundColor: "{colors.lamp}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "14px 38px"
    height: "64px"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
    rounded: "{rounded.pill}"
    padding: "11px 22px"
    height: "48px"
  button-quiet-hover:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
  button-small:
    rounded: "{rounded.pill}"
    padding: "7px 16px"
    height: "40px"
  button-danger:
    backgroundColor: "{colors.miss}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
  button-on-screen:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.pill}"
    padding: "7px 16px"
    height: "40px"
  select-pill:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
    rounded: "{rounded.pill}"
    padding: "0 6px 0 14px"
    height: "44px"
  input-field:
    backgroundColor: "{colors.house}"
    textColor: "{colors.fg}"
    typography: "{typography.body}"
    rounded: "{rounded.field}"
    padding: "11px 14px"
    height: "48px"
  chip-cue:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
    typography: "{typography.caption}"
    rounded: "{rounded.pill}"
    padding: "6px 13px"
    height: "40px"
  panel:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
    rounded: "{rounded.panel}"
    padding: "22px"
  scrim:
    backgroundColor: "{colors.paper-dim}"
    textColor: "{colors.ink}"
    rounded: "{rounded.screen}"
    padding: "22px 26px 96px"
  scrim-listening:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  scrim-error:
    backgroundColor: "{colors.ash}"
    textColor: "{colors.ink}"
  stamp:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
    typography: "{typography.stamp}"
    rounded: "{rounded.stamp}"
    padding: "2px 9px"
  stamp-wrong:
    textColor: "{colors.miss}"
---

# Design System: Tink

## Overview

**Creative North Star: "Silhouette Theatre"**

Tink is a cut-paper shadow play in the manner of Lotte Reiniger. The page is a dark, warm theatre house, and in it hangs one backlit paper screen. The person's sound is the lamp behind that screen: a soft hotspot, drawn every frame from the microphone level, swells with the voice and blooms when Tink understands. What the sound means is the shadow it casts: the phrase appears on the paper in black silhouette type, above a cut-paper London skyline that runs along the screen's foot.

Everything that is operated lives in the house around the screen and stays quiet: hairline frames, transparent panels, warm off-white text. All emphasis is spent on the screen. Six film tints, one per taught sound, are the only saturated colour in the system, and each tint follows its sound everywhere it appears: its list swatch, its ranking bar, its log stamp, and the whole screen when that sound is heard. While Tink speaks, everything outside the screen dims to 38% so the room watches the shadow. Density is moderate and calm; targets are large; the screen is built to be read across a room in present mode.

The world refuses the category default of a dark dashboard with a glowing accent orb and a result card. The result is not a card; it is the screen itself changing colour.

**Key Characteristics:**
- One lit surface (the scrim) on a theatre-black house; nothing else is filled.
- Silhouette ink (the house black) is the only text colour on the screen.
- One film tint per taught sound, assigned by teaching order, carried across every view of that sound.
- Live, procedural light behind the paper, never a static gradient.
- Hairline structure in the house; depth only on the screen.
- A display serif for what is said, a hyperlegible sans for everything operated.

## Colors

A warm, near-monochrome house of browned blacks and paper creams, with one lamp amber for action and six film tints that belong to sounds rather than to the interface.

### Primary
- **Lamp Amber** (`lamp`): the primary action fill (Listen, Record take, Right), the focus ring, the text caret, form accent, selection highlight, pressed and checked states (as a 14% wash), and the Listen ring pulse. It is the lamp of the theatre; it marks what to press.

### Secondary
- **Film Tints** (`tint-0` amber, `tint-1` teal, `tint-2` rose, `tint-3` violet, `tint-4` leaf, `tint-5` sky): assigned to taught sounds in teaching order, cycling after six. A sound's tint fills its swatch in Sounds Tink knows, washes its list row at 16% when lit, fills its ranking bar and swatch, borders its log stamp and its yes/no role chip, and becomes the whole screen colour when that sound is heard. Silhouette ink reads on every tint at 6:1 or better.

### Tertiary
- **Miss Coral** (`miss`): wrong and error only. The Wrong stamp, inline form errors, the forget hover, the Record take button while recording, and danger buttons.

### Neutral
- **Theatre House** (`house`): page background; also the inset ground of inputs and the packet code block.
- **Ember Surface** (`surface`): dropdown option background only.
- **Hairline** (`line`): the 1px frames of panels, the top bar rule, list and log dividers, passive chips.
- **Control Edge** (`control-line`): the edge of anything operable in the house (quiet buttons, select, inputs, cue chips, empty take dots, scrollbar). Holds 4.3:1 on surface and 4.7:1 on house.
- **House Light** (`fg`): primary text in the house.
- **Programme Grey** (`muted`): secondary text, hints, help, timestamps, empty states (8.6:1 on house).
- **Lit Paper** (`paper`): the screen while Tink listens; also the text colour of ink buttons and active chips on the screen.
- **Unlit Paper** (`paper-dim`): the screen at rest and after a miss: the same paper with the lamp off.
- **Silhouette Ink** (`ink`): every mark on the screen (phrase, status, level bars, skyline, ranking outlines, guide rules at 35%) and the text of every lamp or tint filled button. Same value as the house: the shadow is cut from the dark around it.
- **Ash** (`ash`): the screen when Tink is offline or errored, and the tint of a sound that is no longer known.

### Named Rules
**The One Screen Rule.** Only the scrim is filled. Panels, top bar, and controls in the house are transparent with hairline edges; if a new surface wants a fill, it belongs on the screen or it does not get one.

**The Tint Follows the Sound Rule.** A film tint is owned by a sound, never by a component or a state. Every place a sound appears carries its tint; nothing else may use a tint.

**The Ink Only Rule.** On the paper, text and marks are silhouette ink. Colour on the screen comes from the paper itself changing, not from coloured text.

## Typography

**Display Font:** Gloock (with Bodoni 72, Georgia)
**Body Font:** Atkinson Hyperlegible (with Segoe UI, system-ui)
**Label/Mono Font:** JetBrains Mono (with ui-monospace, Menlo), tabular figures

**Character:** Gloock is a high-contrast display serif with the cut, theatrical edge of a title card; it carries only what is said. Atkinson Hyperlegible is built for low vision and carries everything that is operated, so the voice of the person and the voice of the tool never blur.

### Hierarchy
- **Display** (400, clamp(40px, 5.2vw, 68px), 1.06): the phrase on the screen, balanced, capped at 17ch. Arrives with a 0.55s "cast" from blur into focus.
- **Display, present** (400, clamp(56px, 8vw, 140px), 1.06): the same phrase in present mode, capped at 18ch.
- **Headline** (400, 40px, 1; 34px on phones): the Tink wordmark.
- **Numeral** (400, 30px, 1): step numbers in the first-run guide, set in the display face.
- **Title** (700, 19px): panel headings in the house.
- **Body** (400, 17px, 1.55): base text everywhere in the house.
- **Body on screen** (400, 16.5px): the "why" line under the phrase, ink at 82%, capped at 54ch.
- **Label** (700, 15px): field labels, legends, the screen status line.
- **Caption** (400, 14 to 14.5px): hints, help, notes (62ch max), chips.
- **Stamp** (700, 12.5px, 0.04em): log stamps.
- **Mono** (400, 12.5 to 13.5px, tabular): probabilities, confidence, log times, the LLM packet.

### Named Rules
**The Two Voices Rule.** Gloock is for what Tink says and the wordmark only; it never labels a control. Atkinson Hyperlegible is for everything a hand operates.

**The Numbers Are Proof Rule.** Any number a viewer might compare (probabilities, times, confidence, the packet) is set in mono with tabular figures.

## Layout

A centred container (max 1340px, 18px 22px 56px padding) holds a top bar, then a two-column grid: the stage at 1.6fr on the left and a side column of at least 330px on the right, 22px gaps throughout. Under the stage, a two-column lower row holds the log and the packet panels. The side column carries Sounds Tink knows above Teach a sound.

The stage is the screen (min 560px tall, 22px 26px 96px padding, the bottom padding clearing the skyline) followed by a centred pad of controls in the house: Listen, Say it again, Mute, then toggles and a hint.

Spacing runs on a small, even rhythm: 6, 8, 12, 14, 18, 22px. 22px separates panels and columns and pads panels; 14px separates items inside a panel; 12px separates controls in a row.

Responsive: below 980px the main grid stacks to one column; below 700px the lower row stacks; below 560px the container gutter drops to 14px, the screen loses its minimum height and pads 18px 16px 72px, the skyline drops to 60px, the guide becomes a single column, button labels on icon buttons hide, and the action row becomes a two-column grid with Listen spanning both.

Present mode (key P) removes the side column, the lower row, the tagline, model chip, hints, toggles, and notes; the container goes full width (14px 28px 18px); the screen fills the viewport less 210px with a 128px skyline, and type scales by viewport.

**The Dim the House Rule.** While Tink speaks, the top bar, side, lower row, and control pad fade to 38% opacity over 0.4s. The screen never dims.

## Elevation & Depth

The house is flat: no shadows, no fills, structure by 1px hairlines alone. Depth exists only on the screen, and it is light rather than shadow: a canvas behind the phrase draws a warm radial hotspot whose radius and opacity follow the live microphone level, eases toward a higher resting glow while listening or heard, blooms on a match and decays, and darkens toward the edges like a lamp through paper. A faint fibre texture (ink at 7%) is generated once per size and laid over it. With reduced motion, the light is drawn once and does not animate.

### Shadow Vocabulary
- **Screen hang** (`box-shadow: 0 30px 60px -30px rgba(0,0,0,.7), inset 0 0 0 1px rgba(21,17,14,.25)`): the single drop shadow in the system, under the scrim, plus an inked inner edge. Nothing else casts.
- **Focus glow on fields** (`box-shadow: 0 0 0 3px` lamp at 35%): input focus in the house.

### Named Rules
**The Light Not Shadow Rule.** The screen gains depth from the live lamp behind it, never from layered cards or stacked shadows. One drop shadow exists, and it belongs to the screen.

## Shapes

Soft, cut-paper corners on a clear radius ladder: the screen at 22px, panels at 16px, fields and the packet block at 12px, list rows, video, and small icon targets at 10px, stamps at 6px, swatches and bars at 4 to 5px, and every button, select, and chip as a full pill. The one hard silhouette is the skyline: a single filled SVG path in ink along the screen's foot, a cut-out skyline with a tower and a moon hole, clipped by the screen's rounded corners. Stamps sit rotated -2 degrees, like rubber stamps on a programme. Ranking swatches and bars carry a 1.5 to 2px ink outline so tinted fills read as cut shapes on paper.

## Components

### Buttons
- **Shape:** full pill (999px), minimum height 48px, 8px icon gap, Phosphor icons inline.
- **Primary:** lamp fill, ink text, 700 weight, 11px 22px padding. Press settles 1px down at 98% scale over 0.15s.
- **Listen:** the primary at 64px tall, 14px 38px padding, 20px text. While listening, a 2px lamp ring pulses outward from 6px outside the button (1.4s loop).
- **Quiet:** transparent, house light text, 1px control edge; hover turns the edge lamp; pressed (aria-pressed) adds a 14% lamp wash with a lamp edge.
- **Small:** 40px tall, 7px 16px padding, 15px text.
- **On screen:** inside the screen, primary becomes ink fill with paper text, and quiet becomes transparent with an ink edge and ink text.
- **Danger:** coral fill, ink text.
- **Disabled:** 45% opacity, default cursor.
- **Link button on screen:** ink, 700, underlined with 3px offset, 36px minimum height.

### Chips
- **Style:** pill, 13.5 to 14.5px. Passive cue chips are hairline and muted; on the screen they become ink-outlined (45%) ink text.
- **State:** a selectable cue chip is a 40px pill with a control edge; checked turns the edge lamp with a 14% lamp wash. A detected live cue on the screen fills ink with paper text. A yes/no role chip takes its sound's tint as its edge.
- **Model chip:** a muted hairline pill in the top bar naming the model.

### Cards / Containers
- **Corner Style:** 16px.
- **Background:** transparent on the house.
- **Shadow Strategy:** none (see Elevation & Depth).
- **Border:** 1px hairline.
- **Internal Padding:** 22px, 14px between children. Heading row: title left, muted small or a lamp icon right.

### Inputs / Fields
- **Style:** 48px tall, 12px radius, house ground, control edge, 11px 14px padding; label above in 700 15px; help below in 14px muted; "Optional" as a 400 muted 13.5px suffix.
- **Focus:** edge turns lamp with a 3px 35% lamp glow; no outline.
- **Error / Disabled:** errors in coral, 14.5px, under the form with role alert; readonly fields at 75% opacity.
- **Select:** a 44px pill with a control edge, a leading icon, and a drawn chevron; focus draws the 3px lamp outline around the whole pill.

### Navigation
A single top bar, not a nav: wordmark and muted tagline left on a baseline; language select, model chip, and a quiet Present toggle right; a hairline rule below, 64px minimum. Wraps on narrow screens.

### The Screen (signature)
The scrim holds the whole result. States set its paper: unlit paper at rest and after a miss, lit paper while listening, the heard sound's film tint when matched (transition 0.7s), ash on error. Top left, a status line in ink with five level bars that scale with the voice (inner bars at a lower range). Centre, the phrase in display ink, the why line, then either the first-run guide or the result: a taught-phrase row, the ranking, and the Right / Wrong feedback. The camera preview, when on, sits top right as a 132px mirrored frame with a 2px ink border and live cue chips.

### Ranking
A list up to 460px wide (720px in present mode). Each row: a 14px tinted swatch with a 2px ink edge, the sound name, a mono probability, and an 8px tinted bar across the full row with a 1.5px ink inner edge that grows from the left over 0.6s. The leading match inverts to solid ink swatch and bar. "Didn't catch it" (unknown) is a 30% ink bar with no edge.

### First-run Guide
A real three-step sequence, so it is numbered: three columns split by 35% ink hairlines under a 35% ink top rule, each step a display-face numeral, a 700 label, and an 80% ink caption. A finished step's numeral is replaced by a check. Stacks to one column below 560px.

### Sound List
Rows divided by hairlines: an 18px tint swatch, the sound name in 700, its phrase in muted 15px, cue chips beneath, take dots, and a 40px forget control that turns coral on hover. The row just heard is washed in its tint at 16%.

### Stamped Log
Rows on a 70px / 1fr / auto grid: mono time, the spoken text, and a stamp. A stamp is a 1.5px tint-edged label rotated -2 degrees; a Wrong stamp is coral, and its text is struck through with a 2px line and muted. Entries are never removed.

### Packet
A details disclosure in muted 14.5px; open, it shows a mono 12.5px block on house ground with a hairline edge, 12px radius, 14px padding, scrolling past 260px.

## Do's and Don'ts

### Do:
- **Do** give every new taught-sound view its sound's film tint, assigned by teaching order from the six tints.
- **Do** keep the screen the only filled surface; build house panels as transparent frames with a 1px hairline and 16px corners.
- **Do** set every mark on the screen in silhouette ink, including buttons, which invert to ink fill with paper text.
- **Do** keep operable targets at 48px (64px for Listen; 40px only for small and icon controls) and edge them with the control edge so they hold 3:1 in the house.
- **Do** set compared numbers in mono with tabular figures.
- **Do** honor reduced motion by removing every animation and transition and drawing the screen light once.
- **Do** raise secondary text to full strength and hairlines to control strength under increased contrast.

### Don't:
- **Don't** use a film tint for a button, a state, or decoration; tints belong to sounds.
- **Don't** put the result in a card; the screen changing colour is the result.
- **Don't** add drop shadows to panels or controls; the screen hang is the only one.
- **Don't** set controls or labels in Gloock, or the phrase in the sans.
- **Don't** replace the live lamp behind the screen with a static gradient or a glowing orb.
- **Don't** remove a wrong turn from the log; strike it and stamp it Wrong.
