---
name: Tink
description: White paper, slate ink, and deep teal. A quiet editorial page around one screen that shows what a sound means.
colors:
  house: "#fbfbf8"
  surface: "#ffffff"
  surface-2: "#f1f2ee"
  line: "#e3e5df"
  grid: "#e6e9e3"
  control-line: "#8d969a"
  fg: "#16232b"
  muted: "#55626a"
  ink: "#16232b"
  on-ink: "#fbfbf8"
  teal: "#0f766e"
  teal-deep: "#0b5d57"
  teal-text: "#0d6b64"
  light: "#0f766e"
  on-light: "#ffffff"
  hover: "#0d6b64"
  spark: "#b45309"
  spark-soft: "#f5b942"
  miss: "#a8382f"
  alarm: "#f2605a"
  alarm-light: "#f7b8b0"
  paper: "#ffffff"
  paper-dim: "#f1f2ee"
  ash: "#e4e4df"
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

Tink looks like a clinical research page, not a gadget. One white-ish sheet, deep slate type, thin cool rules, and deep teal as the one accent: primary buttons, links, the waveform, and the current page. Teal reads calm and trustworthy, which fits a tool people lean on to be understood. A warm amber spark is a small touch only, for the moment Tink understands. The layout is borrowed from editorial health sites: a fixed side rail, a large serif headline, a faded grid behind the hero, and content split into columns by hairlines rather than boxed into cards.

The only colour that carries meaning is the set of six soft tints, one per taught sound. Each tint follows its sound everywhere: list swatch, ranking bar, log stamp, and the whole screen when that sound is heard. All six are light enough that ink type on them stays readable.

## Colors

All colours live as custom properties on `:root` in `web/style.css`. Nothing else in the CSS uses a raw hex value; translucent ink is mixed from `--ink` with `color-mix`. The only raw values outside the CSS are the theme colour in `web/index.html` and the canvas glow in `web/app.js`, which match the tokens.

- **House / surface**: `#fbfbf8` white paper page, `#ffffff` panels and the specimen card, `#f1f2ee` pressed fills and the screen at rest.
- **Ink**: `#16232b`, a deep slate near-black, for text and the buttons on the lit screen. 15.5:1 on the page. Type on an ink, teal or danger fill is paper (`--on-ink`).
- **Muted**: `#55626a` for body copy under headings and all secondary text. 6.1:1 on the page, 5.6:1 on `#f1f2ee`.
- **Rules**: `#e3e5df` hairlines between columns, rows, and panels; `#e6e9e3` for the faded hero grid. A full ink rule marks the top of a numbered list.
- **Teal**: `#0f766e` is the fill for primary controls (Listen, Load the model), always with white type (5.5:1). It deepens to `#0b5d57` on hover (7.7:1). As type (links on hover, current nav item, headline letters under the cursor) it uses `#0d6b64` (`--teal-text`, 5.7:1). The specimen waveform, the live dot, the mic level bars, and link underlines are teal.
- **Amber spark**: `#b45309` for focus rings and the Measured badge (4.8:1); `#f5b942` only in the hero glow and, softened, in text selection. It never fills a control and never carries meaning.
- **Miss**: `#a8382f` for wrong stamps, errors, and the record and danger fills (paper on it 6.2:1).
- **Tints**: the six sound tints are unchanged except `tint-0`, now `#f3d58a` butter gold, so the first sound stays distinct from the pale screen at rest.
- **Alarm**: `#f2605a` for urgent sounds only (ink on it 5.0:1), pulsing to `#f7b8b0`. It is warm red against a cool teal accent, so urgent never reads as an accent.

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
- **Panels**: white (`#ffffff`), 1px rule, 4px corners, Spectral titles, mono labels on the right.
- **The screen**: a flat pale sheet on a faint 28px grid. Its background takes the heard sound's tint. A canvas draws a soft hotspot from the mic level.
- **Specimen card**: the hero figure, a white card on the faded grid showing a waveform, the sound name, and the phrase. A soft teal to amber radial glow sits behind it.

## Motion

- Headline letters (`data-letters`) turn teal one at a time under the cursor, only on devices that hover.
- Sections fade up 16px once as they scroll in.
- All motion is off under `prefers-reduced-motion`, except the mic level, which is information.

## Do's and Don'ts

Do: keep emphasis in type size and rules, not colour. Keep teal for primary actions, links, and the waveform, and use `--teal-text` whenever teal is small type. Keep amber to focus, selection, one badge, and the hero glow. Say real numbers (27% vs 19% chance) in plain words.

Don't: add gradients, glows, pills, or shadows beyond the glow and shadow behind the specimen card. Don't put ink type on teal; use white. Don't let amber spread into fills or text. Don't put saturated colour on the screen; tints stay soft so ink stays readable.
