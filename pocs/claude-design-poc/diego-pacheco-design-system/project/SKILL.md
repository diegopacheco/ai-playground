---
name: diego-pacheco-design
description: Use this skill to generate well-branded interfaces and assets for the Diego Pacheco engineering portfolio — `https://diegopacheco.github.io/`. Suitable for production code, throwaway prototypes, additional portfolio pages, book/talk microsites, and slide decks that should match the existing look. Contains essential design guidelines, colors, type, fonts, assets, and a UI kit of React components.
user-invocable: true
---

# Diego Pacheco — Design System

This skill packages the design system extracted from Diego Pacheco's personal engineering portfolio. The visual identity is a **light slate-blue page with a dark→cobalt gradient hero**, **white rounded cards with soft shadows**, **Inter typography**, and **heavy use of emoji** as section icons and project markers.

## How to use this skill

1. **Read `README.md`** first — it has the full content fundamentals, visual foundations, and iconography rules.
2. **Use `colors_and_type.css`** as the source of truth for tokens. Either link it directly or copy the `:root` block.
3. **Pull components from `ui_kits/portfolio/`** when you need realistic hero / bio / socials / books-carousel / lectures / list patterns. The class names match the source site so you can compose them freely.
4. **Copy assets from `assets/`** — never reference external URLs from these files for production code. (Tech-language icons are an exception: they're loaded from third-party CDNs on the live site.)
5. **Match the voice** — first-person, dense, emoji-prefixed sections, leading with humanity ("Father" before "Principal Software Architect").

## Output conventions

- **HTML artifacts** (slides, throwaway prototypes, mocks): copy the relevant assets into the output folder and inline / link `colors_and_type.css`. Build static HTML files that the user can open directly.
- **Production code**: read this folder as reference. Lift token values, type scale, shadow/radius scales, and component patterns directly. Do not rebuild a different system.
- **Sub-pages of the portfolio**: keep the outer shell (`.site-shell`, `.card`, hero gradient) and add new sections following the same `card + emoji-headed h3 + content` pattern.

## When invoked without other guidance

Ask the user what they want to build (e.g. *"a new page for the portfolio"*, *"a talk-archive microsite"*, *"slides in this brand"*, *"a launch page for the next book"*). Ask follow-up questions about audience, length, and whether they need any new sections that don't exist on the live site. Then act as an expert designer who outputs HTML artifacts or production code as appropriate.

## Key references

| File | What's inside |
|---|---|
| `README.md` | Full system documentation — content tone, palette, type, motion, iconography, caveats. |
| `colors_and_type.css` | All tokens + semantic CSS + drop-in `.dp-*` utility classes. |
| `assets/` | Profile photo, book covers, 13 social-platform PNGs. |
| `fonts/README.md` | Inter (Google Fonts) loading + substitution notes. |
| `preview/` | 25 reference cards showing every primitive in isolation. |
| `ui_kits/portfolio/` | React (JSX) recreation with `Hero` / `BioCard` / `SocialsGrid` / `BooksCarousel` / `SideProjectStack` / `MiniCardGrid` / `LecturesTable` / `ListCard`. |
| `_research/index.html` | Snapshot of the live upstream source (verified copy of `diegopacheco.github.io`'s `index.html`). |

## Non-negotiables

- **Inter only.** No alternative typefaces unless explicitly requested.
- **No purple-blue tropes, no emoji-cards, no left-border accents.** Stick to the white-card-on-slate-page composition.
- **Emoji is part of the brand.** Don't strip it from section headings, project names, or callouts.
- **Use the real social/brand PNGs from `assets/`.** Don't redraw logos as SVG.
- **Hover effects always lift up and scale assertively** (translateY −6 px, scale 1.05–1.45). Plain `ease`, 0.2–0.35 s.
