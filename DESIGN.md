---
name: Tink
description: Butter cream and ink. A quiet editorial page around one screen that shows what a sound means.
colors:
  house: "#fff6dc"
  surface: "#fffbee"
  surface-2: "#fbefc9"
  line: "#ecdcb0"
  grid: "#e6d3a3"
  control-line: "#bfa97c"
  fg: "#2a1f2d"
  muted: "#5e5060"
  ink: "#2a1f2d"
  on-ink: "#fff6dc"
  coral: "#ff7f50"
  coral-deep: "#e8643a"
  coral-text: "#b3401a"
  light: "#ff7f50"
  on-light: "#2a1f2d"
  hover: "#b3401a"
  galaxy: "#5b2a86"
  galaxy-soft: "#7b4bc4"
  miss: "#a8382f"
  alarm: "#f2605a"
  alarm-light: "#f7b8b0"
  paper: "#fffcf2"
  paper-dim: "#fbefc9"
  ash: "#e3dbc8"
  tint-0: "#f3d58a"
  tint-1: "#cfe0c8"
  tint-2: "#efcdc4"
  tint-3: "#cddcea"
  tint-4: "#e0d7ea"
  tint-5: "#dde6bd"
typography:
  display: { fontFamily: "Spectral, Georgia, serif", fontSize: "clamp(48px, 5vw, 72px)", fontWeight: 400, lineHeight: 1.05, letterSpacing: "-0.02em" }
  headline: { fontFamily: "Spectral, Georgia, serif", fontSize: "clamp(30px, 3vw, 40px)", fontWeight: 400, lineHeight: 1.12, letterSpacing: "-0.01em" }
  title: { fontFamily: "Spectral, Georgia, serif", fontSize: "22px", fontWeight: 400, lineHeight: 1.25 }
  body: { fontFamily: "Inter Tight, system-ui, sans-serif", fontSize: "16-17px", fontWeight: 400, lineHeight: 1.65 }
  eyebrow: { fontFamily: "IBM Plex Mono, monospace", fontSize: "11-12px", fontWeight: 500, letterSpacing: "0.12em", textTransform: uppercase }
radius: { control: "4px", chip: "2px" }
---

# Design System: Tink

## Overview

Tink looks like a clinical research page, not a gadget. One butter cream sheet, deep plum type, thin warm rules, and coral orange as the one accent: primary buttons, link hover, and small highlights. Galactic purple is a small touch only. The layout is borrowed from editorial health sites: a fixed side rail, a large serif headline, a faded grid behind the hero, and content split into columns by hairlines rather than boxed into cards.

The only colour that carries meaning is the set of six soft tints, one per taught sound. Each tint follows its sound everywhere: list swatch, ranking bar, log stamp, and the whole screen when that sound is heard. All six are light enough that ink type on them stays readable.

## Colors

All colours live as custom properties on `:root` in `web/style.css`. Nothing else in the CSS uses a raw hex value; translucent ink is mixed from `--ink` with `color-mix`. The only raw values outside the CSS are the theme colour in `web/index.html` and the canvas glow in `web/app.js`, which match the tokens.

- **House / surface**: `#fff6dc` butter cream page, `#fffbee` panels and the specimen card, `#fbefc9` pressed fills and the screen at rest.
- **Ink**: `#2a1f2d`, a deep plum near-black, for text, the waveform, and the buttons on the lit screen. 14.6:1 on the page. Type on an ink or danger fill is cream (`--on-ink`).
- **Muted**: `#5e5060` for body copy under headings and all secondary text. 7.0:1 on the page, 6.5:1 on `#fbefc9`.
- **Rules**: `#ecdcb0` hairlines between columns, rows, and panels; `#e6d3a3` for the faded hero grid. A full ink rule marks the top of a numbered list.
- **Coral**: `#ff7f50` is the fill for primary controls (Listen, Load the model), always with ink type (6.3:1). It deepens to `#e8643a` on hover (ink on it 4.7:1). Coral is too light for type on cream, so links, nav items, and headline letters turn `#b3401a` (`--coral-text`, 5.3:1) under the cursor. The live dot on the specimen card is coral.
- **Galactic purple**: `#5b2a86` for focus rings, text selection, and the Measured badge; `#7b4bc4` only inside the soft glow behind the hero specimen card. It never fills a control and never carries meaning.
- **Miss**: `#a8382f` for wrong stamps, errors, and the record and danger fills (cream on it 6.0:1).
- **Tints**: the six sound tints are unchanged except `tint-0`, now `#f3d58a` butter gold, so the first sound stays distinct from the cream screen at rest.
- **Alarm**: `#f2605a` for urgent sounds only (ink on it 5.0:1), pulsing to `#f7b8b0`. It is redder than coral so urgent never reads as an accent.

## Typography

- **Spectral** for anything that is said or titled: headlines, the heard phrase, panel titles.
- **Inter Tight** for body and controls.
- **IBM Plex Mono**, uppercase and tracked, for eyebrows, step numbers, timestamps, stamps, and confidence numbers.

## Layout

- Desktop: a 72px fixed rail on the left. Its labels sit rotated until the rail is hovered, then widen to 172px and turn upright. Content is max 1280px with 48px gutters.
- Below 1024px the rail becomes a 56px top bar with dot-separated links; gutters drop to 24px, then 16px inside the app on phones.
- Sections use a 12-column grid (hero 6 + 6, privacy 5 + 6) and lots of vertical space (96-160px).
- Groups of three or four are columns divided by vertical hairlines that turn into horizontal rules on phones.

## Components

- **Links**: underlined text with a trailing arrow that slides 4px on hover. The landing page has no filled buttons.
- **Buttons (app)**: 4px corners, ink fill or a 1px outline (`quiet`). Listen is the largest target.
- **Panels**: cream white (`#fffbee`), 1px rule, 4px corners, Spectral titles, mono labels on the right.
- **The screen**: a flat cream sheet on a faint 28px grid. Its background takes the heard sound's tint. A canvas draws a soft hotspot from the mic level.
- **Specimen card**: the hero figure, a cream white card on the faded grid showing a waveform, the sound name, and the phrase. A soft purple to coral radial glow sits behind it.

## Motion

- Headline letters (`data-letters`) turn coral one at a time under the cursor, only on devices that hover.
- Sections fade up 16px once as they scroll in.
- All motion is off under `prefers-reduced-motion`, except the mic level, which is information.

## Do's and Don'ts

Do: keep emphasis in type size and rules, not colour. Keep coral for primary actions and hover, and use `--coral-text` whenever coral is type. Keep purple to focus, selection, one badge, and the hero glow. Say real numbers (27% vs 19% chance) in plain words.

Don't: add gradients, glows, pills, or shadows beyond the glow and shadow behind the specimen card. Don't put cream or white type on coral; it fails contrast. Don't let purple spread into fills or text. Don't put saturated colour on the screen; tints stay soft so ink stays readable.
