# Design System — qa2pw

The single source of truth for visual decisions in the qa2pw playground.
Always read this before writing any UI code. Do not deviate without explicit user approval.

## Product Context

- **What this is:** A hosted web playground that converts plain-English QA test cases into clean Playwright `.spec.ts` files by driving a real browser with Claude.
- **Who it's for:** QA engineers, developers writing tests, technical PMs who want regression coverage without writing selectors.
- **Space:** Developer tools. Peers in mind: Linear, Vercel dashboard, Railway, Stagehand playground, Playwright codegen UI.
- **Project type:** Single-page web app where the playground IS the landing page. No marketing site, no separate dashboard.

## Memorable Thing

> "Watch Claude write a Playwright test in front of you, then download it."

Every design decision below serves that one sentence. The screencast pane has to feel inviting before the run starts and alive during it. The script pane has to feel like a finished artifact you'd commit to git. The form has to feel like a one-click invitation, not a wall.

## Aesthetic Direction

- **Direction:** Industrial / Utilitarian with editorial restraint. Linear / Vercel / Railway register, not Notion / Stripe.
- **Decoration level:** Minimal. Typography and one accent color do all the work. No textures, no gradients, no decorative borders, no blobs.
- **Mood:** Calm, confident, technical. The page reads like a precision instrument, not a marketing brochure.
- **Reference register:** Linear's app surface, Vercel's dashboards, Railway's project pages, GitHub's code review surface. NOT Notion (too card-y), NOT Stripe (too gradient-y), NOT Slack (too chrome-y).

## Typography

| Role | Face | Weight | Loading |
|---|---|---|---|
| Display (wordmark, h1) | JetBrains Mono | 700 | Google Fonts or self-host |
| UI body, headings, form | Inter Tight | 400 / 500 / 600 | Google Fonts or self-host |
| Code preview pane | JetBrains Mono | 400 / 500 | Same load as wordmark |
| URL input | JetBrains Mono | 400 | Same load |

Never use `system-ui`, `-apple-system`, Inter (the regular non-Tight version is the slop default), Roboto, Arial, or Helvetica anywhere.

### Type scale

Modular, base 16px. Tailwind-compatible names in parens.

| Token | px / rem | Use |
|---|---|---|
| `text-xs` | 12 / 0.75 | Captions, helper text, table data labels |
| `text-sm` | 14 / 0.875 | Secondary UI, status announcer, microcopy |
| `text-base` | 16 / 1 | Body text, form inputs, button labels (minimum) |
| `text-lg` | 18 / 1.125 | Tagline, section labels |
| `text-xl` | 20 / 1.25 | Pane titles |
| `text-2xl` | 24 / 1.5 | Step caption (the "watch it think" line) |
| `text-3xl` | 30 / 1.875 | Wordmark + (reserved) |
| `text-4xl` | 36 / 2.25 | Reserved (no current use; first-paint hierarchy works without it) |

Body text minimum is 16px. Never go below 14px even for secondary text — the playground is read for minutes at a time.

Line-height: 1.5 for body, 1.2 for headings/wordmark, 1.6 for code pane.

## Color

Restrained palette. One accent. Light mode only in v1.

### Tokens (CSS variables)

```css
:root {
  /* Surfaces */
  --bg:           #FAFAFA;  /* near-white page background */
  --surface:      #FFFFFF;  /* panes, inputs, the script preview area */
  --surface-2:   #F4F4F5;  /* hover/active states, disabled buttons, code pane background */
  --divider:      #E4E4E7;  /* 1px lines between panes, under headers */

  /* Type */
  --text:         #18181B;  /* body, primary text */
  --text-muted:   #71717A;  /* helper text, captions, placeholder */
  --text-faint:   #A1A1AA;  /* the faint browser-chrome outline, empty-pane hint copy */

  /* Accent — amber */
  --accent-fill:  #F59E0B;  /* solid Generate button fill, ✓ overlay, active focus rings */
  --accent-text:  #B45309;  /* amber text on white surfaces (passes WCAG AA 4.5:1) */
  --accent-soft:  #FEF3C7;  /* partial-on-timeout banner background, attestation hint */

  /* Semantic */
  --success:      #15803D;  /* run complete checkmark, green text */
  --success-soft: #DCFCE7;
  --warning:      #B45309;  /* same as accent-text; partial-on-timeout */
  --warning-soft: #FEF3C7;  /* same as accent-soft */
  --error:        #B91C1C;
  --error-soft:   #FEE2E2;

  /* Focus */
  --focus-ring:   #F59E0B;  /* accent-fill at 2px solid */
}
```

### Color rules (binding)

- White text on `--accent-fill` (amber-500) only. Amber text always uses `--accent-text` (amber-700+) — never put amber-500 text on white.
- No purple, indigo, or violet anywhere on the page. Not in error states, not in placeholder strings, not in syntax highlighting, not in defaults.
- No gradients. Solid fills only.
- Surfaces are flat. No shadows on panes, buttons, inputs, or cards. Shadows reserved for overlays only (modals, dropdowns) with a soft `0 4px 12px rgba(0,0,0,0.08)`.

### Dark mode

Not in v1. Add when the playground has 1k+ daily uniques and the issue is asked for.

## Spacing

Base unit 4px. Density: comfortable (not Notion-loose, not Slack-tight).

| Token | px | Use |
|---|---|---|
| `space-1` | 4 | Tight icon-to-label gaps |
| `space-2` | 8 | Within form rows, between adjacent buttons |
| `space-3` | 12 | Between form field rows |
| `space-4` | 16 | Pane internal padding, default gap |
| `space-6` | 24 | Section spacing within a pane |
| `space-8` | 32 | Header-to-content gap |
| `space-12` | 48 | Outer page padding, between major regions |
| `space-16` | 64 | Reserved (no current use at single-page scale) |

## Layout

- **Approach:** Grid-disciplined for app surface (the three panes). No editorial overlap or asymmetry.
- **Page container:** Max-width `1440px` centered with `--space-12` horizontal padding above `1280px` viewports, `--space-8` between.
- **Pane geometry:** Three columns at `1fr 1.4fr 1.2fr` (form | screencast | script). Screencast is the widest because it carries the magic. 1px `--divider` between columns. No outer border on the pane group.
- **Header bar:** 64px tall, sits above panes, contains wordmark (left) + tagline (just right of wordmark). No nav, no buttons, no chrome.
- **Pane internal padding:** `--space-4` top/bottom, `--space-6` left/right.

### Border radius

| Token | px | Use |
|---|---|---|
| `radius-sm` | 4 | Inputs, the script preview area corners |
| `radius-md` | 6 | Buttons, status banners, attestation checkbox surround |
| `radius-lg` | 8 | (Reserved — only if a modal/dropdown is added later) |
| `radius-full` | 9999 | The Generate-button spinner only |

Never use `rounded-xl` (12px) or larger on layout elements. The IDE feel depends on sharp corners.

## Components

### Buttons

| Variant | Use | Style |
|---|---|---|
| Primary (`btn-primary`) | Generate button only | Solid `--accent-fill` background, white text, `radius-md`, 48px tall, `text-base`/500 weight |
| Outline (`btn-outline`) | Download, Continue, Try again | 1px `--text` border, transparent background, `--text` text, `radius-md`, 40px tall |
| Ghost (`btn-ghost`) | Tertiary (clearing forms, copying script) | No border, `--text-muted` text, hover sets `--surface-2` background, 40px tall |

All buttons: `text-base` (16px) labels, JetBrains Mono for command-y labels (e.g., "Generate", "Download"), Inter Tight for prose labels (e.g., "Try again").

### Inputs

- 40px tall, `radius-sm`, 1px `--divider` border, `--surface` background, `--text` text.
- Focus state: 2px `--focus-ring`, no border color change.
- Placeholder text uses `--text-faint`.
- URL input is mono (`JetBrains Mono`), all other inputs are Inter Tight.
- The English-prompt textarea is `200px` min height, `text-base`, line-height 1.5, resizable vertically.

### Status banners

- 48px tall, full-pane width above relevant pane, `radius-md`, `--space-4` internal padding.
- Partial timeout: `--warning-soft` background, `--warning` left 3px stripe, body in `--text`.
- Rate limit: `--error-soft` background, `--error` left 3px stripe.
- Sleeping: full-page takeover, see "Sleeping page" below.

### Code preview pane

- Background `--surface-2`, `text-base` (16px) JetBrains Mono, line-height 1.6.
- Line numbers in `--text-faint` separated by `--space-4` gap.
- Syntax theme: see "Syntax Theme" below.
- Lines fade in (200ms ease-out, no transform) as the run produces them — never typewriter, never jumpy.

### Sleeping page

- Full viewport, `--bg` background, centered content stack.
- Wordmark at top in `text-3xl`.
- Headline: "Playground's resting until midnight UTC." `text-2xl` Inter Tight 500.
- One-line body: "Daily budget hit. Star the repo to get notified when v2 lifts the cap." `text-base` `--text-muted`.
- Single ghost button to GitHub.
- No illustrations, no clock graphic, no marketing.

## Syntax Theme (Code Pane)

Custom Prism/Shiki theme, light, tuned to the amber palette.

```css
.tok-keyword   { color: var(--accent-text); font-weight: 500; }  /* await, function, import, const */
.tok-string    { color: #92400E; }                                /* warm amber-brown */
.tok-comment   { color: var(--text-muted); font-style: italic; }
.tok-function  { color: var(--text); font-weight: 500; }
.tok-variable  { color: var(--text); }
.tok-punct     { color: var(--text-faint); }
.tok-number    { color: var(--text); }
```

Background of code pane: `--surface-2`. Selection: `--accent-soft`.

## Motion

Minimal-functional. Motion exists only to aid comprehension.

| Easing | Curve | Use |
|---|---|---|
| `ease-out` | `cubic-bezier(0.16, 1, 0.3, 1)` | Entrances (frame fade-in, script line append, banner reveal) |
| `ease-in-out` | `cubic-bezier(0.4, 0, 0.2, 1)` | State transitions (button hover, focus rings) |

| Duration | ms | Use |
|---|---|---|
| `dur-micro` | 100 | Button hover, focus ring appear |
| `dur-short` | 200 | Script line fade-in, banner reveal |
| `dur-medium` | 350 | Screencast frame swap |
| `dur-long` | 600 | Reserved (no current use) |

No bounce, no scale transforms, no opacity-and-translate combos. Frames swap by opacity only. The Generate-button spinner is a single 1s linear rotate.

## Accessibility (v1 baseline)

- **Keyboard nav order:** `textarea → URL input → attestation (if visible) → Generate button → Download → Continue (if partial-state visible)`. After run completion, focus moves to Download.
- **Focus rings:** 2px `--focus-ring` solid, no `outline: none` anywhere. Visible on every interactive element.
- **Contrast:** Body text and any amber-colored text uses `--text` or `--accent-text` (both pass WCAG AA 4.5:1 on `--surface` and `--bg`). `--accent-fill` is reserved for solid backgrounds with white text, never for text on white.
- **`aria-live="polite"`:** Single status region announces `"Started"`, `"Step N of 25"`, `"Run complete"`, `"Error: <one-line>"`. Per-step microcopy is visual only — screen reader users would otherwise hear 25 announcements per run.
- **Touch targets:** N/A (desktop-web only in v1).
- **Reduced motion:** `@media (prefers-reduced-motion: reduce)` disables all fade-ins and frame transitions; everything is instant.

## Anti-Slop Rules (binding for the implementer)

These are the patterns that would make qa2pw look AI-generated. None of them ship.

- No purple / violet / indigo gradients or accents.
- No `system-ui` or `-apple-system` as primary or fallback display/body font.
- No `rounded-xl` (12px+) on layout elements. Sharp corners only.
- No 3-column icon-circle feature grids. (We have one column with three panes; that's different.)
- No `text-align: center` on form blocks or pane content. Center only the sleeping page and (optionally) the wordmark+tagline.
- No decorative shadows. Shadows are for overlays only.
- No emoji in headings or as bullet points.
- No "Welcome to qa2pw" hero copy. The tagline + prefilled example is the hero.
- No carousel, no testimonial section, no pricing section (there's no pricing).
- No skeleton-loader libraries. All loading states are spec'd above.
- No animated background gradients. No blob backgrounds. No SVG dividers.

## Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-23 | Initial design system created | Formalized from /plan-design-review locked decisions. Sources: D6 typography, D7 color, D9 brand, D10 syntax theme + baked-in surfaces/anti-slop rules. |
| 2026-05-23 | Light mode only in v1 | User chose light + amber over dark + green in D7. Dark mode deferred to v2 toggle. |
| 2026-05-23 | Desktop web only | User chose no mobile work in D8. Mobile visitors see the desktop layout collapsed. |
| 2026-05-23 | Cardless surfaces, 1px dividers | Baked in during plan-design-review Pass 4. Linear/IDE register, not Notion. |

## Implementer Read-Order

When you sit down to build the web layer:

1. Load Inter Tight (400/500/600) and JetBrains Mono (400/500/700) from Google Fonts or self-host as `@font-face` in `app/layout.tsx`. Set `font-family: var(--font-ui), Inter Tight, sans-serif` and a separate `--font-mono` chain.
2. Drop the CSS variables block above into `app/globals.css`.
3. In `tailwind.config.ts`, extend the color, spacing, fontFamily, and borderRadius scales to match these tokens. Delete every default Tailwind color you won't use (everything except amber/zinc/red/green).
4. Build components in the order: Header (wordmark+tagline), FormPane, ScreencastPane (with empty state), ScriptPane (with empty state), then state overlays (banners, sleeping page).
5. Sanity check every screen against the Anti-Slop Rules before pushing.
