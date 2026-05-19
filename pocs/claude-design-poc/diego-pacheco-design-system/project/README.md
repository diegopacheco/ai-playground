# Diego Pacheco — Design System

A design system extracted from the personal engineering portfolio of **Diego Pacheco**, Principal Software Architect, author, and speaker — `https://diegopacheco.github.io/`.

The site is a single-page, dense, info-rich résumé/portfolio for a 20+ year software architect. It pairs a **dark-to-blue gradient hero** with **soft white cards** on a light slate-blue page, and uses **emoji as section icons** and **tech logos as inline badges** to give a casual, engineer-friendly tone to what is otherwise a fairly formal CV. The aesthetic sits at the intersection of "modern dashboard" and "personal homepage."

This system is intended for building:
- Additional pages of the portfolio (project deep-dives, talk archives, etc.)
- Companion microsites for the books / video series
- Conference / talk slide decks in the same look
- Blog / newsletter visual identity that matches the portfolio

---

## Source material

- **Live site:** https://diegopacheco.github.io/
- **Repo (verified):** `github.com/diegopacheco/diegopacheco.github.io` — single `index.html` (~65 KB, ~1300 lines) with all styles inline, JS for stacked-card rotation, books carousel, and responsive mobile/desktop column switching.
- **Snapshot:** `_research/index.html` — full copy of the live source, captured for reference.
- All visual assets are in `assets/` (book covers, profile, social/iconography PNGs) — copied directly from the upstream repo's `/images/` folder.

> ℹ️ Tech-language icons (Rust, Scala, Zig, Kotlin, Clojure, Haskell, Nim, V) are loaded from third-party CDNs (`slackmojis.com`, `vlang.io`) on the live site and could not be cached locally due to CORS. Reference them by URL when needed, the same way the live site does, or substitute equivalent logos.

---

## What's in this system

| File / folder | Purpose |
|---|---|
| `README.md` | This file — context, fundamentals, foundations, iconography, index. |
| `SKILL.md` | Cross-compatible Agent Skill manifest for use as a downloadable skill. |
| `colors_and_type.css` | All design tokens — color, type, radius, shadow, spacing — plus semantic CSS selectors (`h1`, `p`, `.card`, etc.). |
| `assets/` | Logos, profile photo, book covers, social icons. Copy what you need into projects. |
| `fonts/` | Local fallback note for **Inter** (Google Fonts is the canonical source). |
| `preview/` | Self-contained HTML cards that render in the Design System tab. |
| `ui_kits/portfolio/` | React (JSX) UI kit recreating the portfolio's hero, bio card, socials grid, books carousel, side-project stack, and content sections. `index.html` is a pixel-faithful click-through. |

---

## Content fundamentals

### Voice
First-person, conversational, slightly **résumé-formal**. The bio reads as one long, breathless paragraph — a "let me tell you everything I've done" register that conveys depth-of-experience without polish. Casing is **Title Case for section headings**, sentence case for body. Acronyms are uppercase (SOA, DevOps, AWS, KMS, EKS).

The "I" is everywhere ("I've led", "I have a passion for"). Section labels addressed to the reader are minimal — there is no "About **you**" framing; the whole site is "About **me**."

### Tone
- **Warm and earnest** — "Brazilian software architect, SOA expert…" leads with identity.
- **Self-confident but not slick** — lists certifications, books, talks, dozens of POCs without curation; volume signals dedication.
- **Engineer-to-engineer** — assumes the reader knows what *Karyon*, *Dynomite*, *Ribbon* are.
- **Family-first** — the "About me" list literally starts with **Father** and **Cat's Father** before any title. Keep this energy: humanize first, then list credentials.

### Specific copy patterns

- **Eyebrow tagline** uses bullet-separated roles in uppercase with wide letter-spacing:
  > `PRINCIPAL SOFTWARE ARCHITECT • AUTHOR • SPEAKER • MENTOR • LEADER`
- **Bullet lists** for personal attributes — single-word or short-phrase items, no terminal punctuation:
  > Father · Cat's Father · Principal Software Architect · SOA Expert · DevOps Practitioner · Author · Mentor · Speaker · Leader
- **Project descriptions** are 2–4 words and lean on a parent language/tech:
  > "Java 23 PL" · "Vanilla JS Trello" · "Zig 0.13 vim-like" · "Go Linux terminal" · "Git-like in Kotlin"
- **"Currently" block** starts with 🌱 and runs together: career line, then hobbies, then a blog URL.
- **Section headings** are always prefixed with an emoji:
  > 📝 Tiny Essays · 🥇 Tiny Side Projects · 👨‍💻 Diego Pacheco Bio · 💻 Core skills and expertise · 🧪 Feature POCs · 🔉 Recent Lectures · 📜 Papers · 💯 Certifications · 🤖 Feature AI POCs

### Emoji usage
**Heavy and intentional.** Every section heading carries an emoji prefix; project names start with an emoji; the country flag 🇧🇷 punches up "Brazilian"; 🌱 marks "Currently"; ☕ tags Sun/Java certifications. Emoji is part of the brand — do not strip it.

---

## Visual foundations

### Palette

The site is built from a tight 5-token palette plus a hero gradient. Everything else derives from these.

| Token | Value | Usage |
|---|---|---|
| `--bg` | `#f5f7fb` | Page background — cool, slate-tinted off-white |
| `--card` | `#ffffff` | All card surfaces |
| `--text` | `#1a1f2b` | Body text, headings (near-black with blue undertone) |
| `--muted` | `#5f6b81` | Captions, table headers, secondary labels |
| `--primary` | `#5a67d8` | Links, social-card tint border, focus accents (indigo) |

**Hero gradient:** `linear-gradient(135deg, #373b44, #4286f4)` — dark slate-graphite into bright cobalt. This is the signature visual of the brand. Use it for any "headline" surface.

**Card-on-hero overlay:** `rgba(15, 23, 42, 0.3)` — translucent navy that lets the hero gradient bleed through. Inner side-project / mini-cards then nest on top with a deeper indigo gradient `linear-gradient(165deg, rgba(22, 30, 78, 0.98), rgba(49, 66, 145, 0.98))`.

**Social-card tint:** `rgba(90, 103, 216, 0.08)` background with `rgba(90, 103, 216, 0.2)` border — used for the 13-up icon grid.

### Type
- **Family:** Inter (Google Fonts, weights 400/500/600/700). Sans-serif fallback `'Inter', sans-serif`.
- **Base size:** 16px, line-height 1.6.
- **Scale (fluid):**
  - `h1`: `clamp(1.8rem, 3vw, 2.8rem)` — 28.8 → 44.8 px
  - `h2`: `clamp(1.2rem, 2vw, 1.6rem)` — 19.2 → 25.6 px
  - `h3`: 1.1rem — 17.6 px
  - body: 1rem — 16 px
  - small / mini-labels: 0.65–0.85rem with letter-spacing 0.08–0.3em and uppercase for eyebrow text
- **Weights:** 400 body, 500 links, 600 headings & badge labels, 700 reserved for ultra-emphatic UI.
- **Letter-spacing:** Tight on body. Generous (0.25em–0.3em) on uppercase eyebrows and labels.

### Backgrounds
- Page is **flat color**, never a gradient or texture.
- Only the **hero** uses a gradient. Inside the hero, **nested layers darken** the gradient with translucent navy to create depth.
- No imagery in backgrounds. **No noise / grain / patterns.** Everything is solid surfaces + rounded corners.

### Borders
- Almost **no visible borders** on cards. Depth comes from shadow.
- Exception: **social cards** use `1px solid rgba(90, 103, 216, 0.2)` to keep the icon grid visually defined.
- Side-project cards in the hero have `1px solid rgba(255, 255, 255, 0.3)` to read against the dark gradient.

### Corner radii
| Scale | Value | Use |
|---|---|---|
| `--r-pill` | `999px` | Buttons (Prev / Next), tags |
| `--r-xl` | `24px` | All major cards, hero |
| `--r-lg` | `18px` | Inner nested cards (side projects, books) |
| `--r-md` | `16px` | Small inline panels (mini-card image wells) |
| `--r-sm` | `14px` | Mini-cards |
| `--r-xs` | `12px` | Social cards, small chips |
| `--r-2xs` | `10px` | Mini-card images |

Pattern: **the further nested, the smaller the radius** — 24 → 18 → 16 → 14 → 12 → 10.

### Shadows
A two-tier system: subtle for resting state, dramatic for hero / lift / hover.

- **Card resting:** `0 20px 60px rgba(15, 23, 42, 0.08)` — soft, deep, blue-tinted.
- **Mini-card resting:** `0 10px 24px rgba(13, 20, 50, 0.45)` — punchy, on dark backgrounds only.
- **Side-project stack:** `0 14px 28px rgba(15, 23, 42, 0.45)` resting, `0 20px 45px rgba(15, 23, 42, 0.6)` for the top "active" card.
- **Book card resting:** `0 8px 24px rgba(15, 23, 42, 0.25)`, lifts to `0 16px 32px rgba(15, 23, 42, 0.35)`.
- **Social card hover:** `0 12px 22px rgba(15, 23, 42, 0.15)` — no resting shadow.

### Spacing
The site doesn't expose a numeric scale, but it cleaves to: **6 · 8 · 10 · 12 · 16 · 20 · 24 · 28 · 32 · 38 · 72** px. Card padding is 24px standard, 28–38px on the hero. Section gap is 24px. Internal grid gaps run 6–16px.

### Animation
**Restrained but present.** Everything that moves uses a 0.2–0.35s ease transition.

- **Mini-card hover:** `transform: translateY(-6px) scale(1.2)` + shadow bump + image scales to 1.45 + opacity to 1. Z-index lift so the card pops above neighbors.
- **Social-card hover:** image scales to 1.35; card gets a shadow.
- **Book card hover:** `translateY(-6px) scale(1.05)` + shadow bump + opacity 0.9 → 1.
- **Side-project stack:** auto-rotates every 3s on `setInterval`, pauses on hover. Stacked cards translate down 10px each via `--stack-index` custom property.
- **Books carousel:** smooth `transform: translateX()` on a slider, 0.35s ease. Click-and-drag horizontal scroll.

Easing is always plain `ease` — no springs, no bounce, no custom cubic-bezier. Hover effects always **lift up** (negative Y translation). Image scales for hover are **assertive** (1.2–1.45) — this is a key brand tic.

### Layout rules
- **Outer max-width: none** — the site stretches edge-to-edge with 32px page padding. On mobile (< 1024 or touch device), `body.single-column` collapses everything to a single stack at 16px padding.
- **Card-of-cards structure** — major sections (`.hero`, `.bio-section`, `.socials-column`, `.details-column`) are all `.card`s; nested sub-elements are smaller cards.
- **Main grid:** `260px sidebar · minmax(0, 1fr) main column` for socials + details. Falls back to 2-column then 1-column.
- **AI POC list** is a 4-column grid (down to 2 on tablet, 1 on mobile).
- **Lectures table** is a simple 3-column HTML `<table>` with year column headers.

### Hover & interaction states
- Links: `opacity: 0.85` on hover (no color change). Underline is **off**.
- Buttons: `background: rgba(255,255,255,0.3)` (lighter) — these only appear on the dark hero, so they brighten on hover.
- No explicit `:active` / pressed state — uses default browser behavior.
- **No focus rings styled** — default browser outline (an accessibility gap worth flagging).

### Transparency & blur
- **No `backdrop-filter: blur`.** Depth is created with stacked translucent navy fills, not glass effects.
- Translucency is used heavily on the hero: 0.08, 0.12, 0.2, 0.3, 0.45, 0.6, 0.65, 0.75 — a deep ladder of navy alphas to nest cards-within-cards.

### Imagery color vibe
- **Profile photo** is the only personal photograph. It's circular-cropped, 110px, ringed by a translucent white circle.
- **Book covers** are full color — they bring the warmth and energy. The system lets them be themselves.
- **Tech logos** are mostly bright, glossy, semi-3D PNGs in the Slackmoji style — saturated, playful. They contrast with the otherwise sober color palette and are the brand's "fun" injection.

---

## Iconography

The site uses **three distinct icon systems**, each for a specific role.

### 1. Emoji as section markers (default UI iconography)
Native Unicode emoji is the primary way sections are labeled. Examples:
- 📝 Essays · 🥇 Side Projects · 👨‍💻 Bio · 💻 Skills · 🧪 POCs · 🔉 Lectures · 📜 Papers · 💯 Certifications · 🤖 AI POCs · 🌱 Currently · 🇧🇷 Brazilian
- Project emoji: 🧝🏾‍♂️ Tupi · 🥫 Jello · 📑 Zim · 💻 Gorminator · 😸 kit · 🦀 Shrust · 🕵🏽 Smith · 📟 ZOS · 🎮 Tiny Games
- AI POC emoji: 🏙️ 📄 🦀 🔮 🦙 🌐 ☸️ 📚 📊 🤖 🎮 💬 🏛️ 🎲 📋 🔐 ⚡ 🐺 📖 ⏱️ 🏗️ 🔧

Emoji is rendered by the OS — no custom font loaded. The site **relies on system emoji rendering** (Apple Color Emoji on macOS/iOS, Segoe UI Emoji on Windows, Noto Color Emoji on Android/Linux). This means visual presentation will vary across platforms — that's an accepted trade-off.

### 2. Brand logo PNGs (socials grid)
All 13 social channels use platform-supplied PNG logos at 52×52 within a 12px-radius tinted card. These are checked into `assets/` and should be referenced directly:
- `github-logo.png`, `linkedin-logo.png`, `Blogger_icon.svg.png`, `substack_logo.png`, `medium.png`, `youtube.png`, `X_icon.svg.png`, `bluesky.png`, `HN.png`, `slideshare.png`, `docker.png`, `vimeo2.png`, `amazon-author.png`

These are **the real brand marks of each platform** — do not redraw, do not stylize. Replace cleanly when a platform updates its identity.

### 3. Tech-language Slackmoji (mini-cards for essays / books)
The 8 essay mini-cards use **Slackmoji-style colorful tech logos** loaded from CDNs:
- Rust, Scala, Zig, Kotlin, Clojure, Haskell, Nim — `emojis.slackmojis.com` / `slackmojis.com`
- V — `vlang.io/img/v-logo.png`

These are **third-party hosted** on the live site. For local reproduction, either keep the CDN URLs or substitute equivalent language marks from a permissible source (Simple Icons, devicon, etc.) and **flag the substitution**.

### What is *not* used
- No custom icon font (no Font Awesome, no Material Icons, no Lucide / Heroicons / Feather).
- No inline SVG icons in the source.
- No PNG sprite sheet.

If you need a *new* UI icon (e.g. an arrow, settings cog) and the design system doesn't already provide one, use a system emoji or the Unicode character `↗` (the books-carousel "external link" uses this). The arrow `↗` after the side-project link label is the site's only non-emoji glyph icon — match this pattern.

---

## Index

- **[`SKILL.md`](./SKILL.md)** — Skill manifest. If this folder is downloaded into Claude Code's `~/.claude/skills/`, it becomes invocable.
- **[`colors_and_type.css`](./colors_and_type.css)** — Drop-in CSS file. Imports Inter, declares all tokens, sets up semantic selectors. Include this at the top of every new file you build in this style.
- **[`assets/`](./assets)** — All real visual assets. Profile photo, book covers, 13 social PNGs.
- **[`preview/`](./preview)** — Self-contained design-system reference cards (typography, palette, components, etc.). Render in the Design System tab.
- **[`ui_kits/portfolio/`](./ui_kits/portfolio)** — Portfolio UI recreation. Run `index.html` to see the click-through.
  - `Hero.jsx` · `BioCard.jsx` · `SocialsGrid.jsx` · `BooksCarousel.jsx` · `SideProjectStack.jsx` · `MiniCardGrid.jsx` · `LecturesTable.jsx` · `ListCard.jsx`

---

## Caveats & things to verify

- **Inter is loaded from Google Fonts CDN**, not bundled. If offline use matters, download the WOFF2 files and self-host.
- **Tech-language logos** (Rust, Scala, etc.) are third-party CDN images, not in `assets/`. They could break if the source goes down.
- **`Tech_Resources_logo.png`** from the upstream repo is ~2.9 MB and was intentionally skipped during sync; fetch on demand if you need it.
- **Focus rings are unstyled.** Anything you build on this system should add a visible focus state.
- **Mobile/desktop column switching is JS-driven** (UA sniff + width media query). A modern rebuild should rely on CSS container queries instead.
