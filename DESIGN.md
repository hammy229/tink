---
name: Tink
description: Paper and ink. A quiet editorial page around one screen that shows what a sound means.
colors:
  house: "#fafaf7"
  surface: "#ffffff"
  surface-2: "#f3f3ee"
  line: "#e5e5e0"
  grid: "#d4d4cd"
  control-line: "#b8b8b0"
  fg: "#111111"
  muted: "#555555"
  ink: "#111111"
  on-light: "#fafaf7"
  hover: "#c45c5c"
  miss: "#b4483f"
  alarm: "#e0675b"
  paper: "#ffffff"
  paper-dim: "#f3f3ee"
  ash: "#d9d9d2"
  tint-0: "#eadcae"
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

Tink looks like a clinical research page, not a gadget. One off-white sheet, near-black type, thin grey rules, and a single muted red that appears only on hover. The layout is borrowed from editorial health sites: a fixed side rail, a large serif headline, a faded grid behind the hero, and content split into columns by hairlines rather than boxed into cards.

The only colour that carries meaning is the set of six soft tints, one per taught sound. Each tint follows its sound everywhere: list swatch, ranking bar, log stamp, and the whole screen when that sound is heard. All six are light enough that ink type on them stays readable.

## Colors

- **House / surface**: `#fafaf7` page, `#ffffff` panels and the live screen.
- **Ink**: `#111111` for text and primary buttons. Primary buttons are ink with paper text.
- **Muted**: `#555555` for body copy under headings and all secondary text.
- **Rules**: `#e5e5e0` hairlines between columns, rows, and panels. A full ink rule marks the top of a numbered list.
- **Hover**: `#c45c5c`. Links, nav items, and headline letters turn this colour under the cursor. It is never a fill.
- **Alarm**: `#e0675b` for urgent sounds only.

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
- **Panels**: white, 1px rule, 4px corners, Spectral titles, mono labels on the right.
- **The screen**: a flat sheet on a faint 28px grid. Its background takes the heard sound's tint. A canvas draws a soft hotspot from the mic level.
- **Specimen card**: the hero figure, a white card on the faded grid showing a waveform, the sound name, and the phrase.

## Motion

- Headline letters (`data-letters`) turn red one at a time under the cursor, only on devices that hover.
- Sections fade up 16px once as they scroll in.
- All motion is off under `prefers-reduced-motion`, except the mic level, which is information.

## Do's and Don'ts

Do: keep emphasis in type size and rules, not colour. Keep red for hover. Say real numbers (27% vs 19% chance) in plain words.

Don't: add gradients, glows, pills, or shadows beyond the one under the specimen card. Don't put saturated colour on the screen; tints stay soft so ink stays readable.
