# Portfolio UI Kit — Diego Pacheco

Pixel-faithful React recreation of `https://diegopacheco.github.io/`, split into reusable JSX components for use in mockups, slides, and microsites.

## Run

Open `index.html` directly. It uses Babel-in-browser; no build step. All scripts are pinned to specific CDN versions with integrity hashes.

## Components

| File | What it renders | Notes |
|---|---|---|
| `Hero.jsx` | Profile photo · eyebrow · name · roles list · 1-line bio · the inner hero panel that holds everything below. | Outermost `.hero.card`. Composes `MiniCardGrid`, `SideProjectStack`, `BooksCarousel`. |
| `MiniCardGrid.jsx` | "📝 Tiny Essays" grid — 8 tech-language mini-cards. | Language icons load from third-party CDNs (slackmojis / vlang.io). |
| `SideProjectStack.jsx` | "🥇 Tiny Side Projects" auto-rotating stacked card. | `setInterval` 3 s, pauses on hover. Uses `--stack-index` CSS variable. |
| `BooksCarousel.jsx` | "Published Books & Video Series" overlapping carousel. | Prev/Next buttons; cards overlap by 112 px (margin-left:-112). |
| `BioCard.jsx` | "👨‍💻 Diego Pacheco Bio" long-form paragraph + "🌱 Currently". | Verbatim copy from the live site. |
| `SocialsGrid.jsx` | "Socials" — 3-column grid of 13 platform PNGs. | All icons in `../../assets/`. |
| `LecturesTable.jsx` | "🔉 Recent Lectures" 3-column year table. | Plain HTML `<table>`. |
| `ListCard.jsx` | Generic emoji-headed list. | Used for Skills, POCs, Papers, Certifications, AI POCs. Pass `listClass` to switch grid layout. |
| `App.jsx` | Page composition + data for the list cards. | All link URLs preserved. |

## What's intentionally simplified

This is a UI recreation, not a production fork. The following live behaviors are reproduced *visually* but not interactively:

- Books carousel — `Prev` / `Next` advance the slide but the drag-to-scroll and arrow-key handlers from the original are skipped.
- Tiny Essays mini-cards — the original rotates items on touchscreen drag; here, only the grid layout + hover behavior is preserved.
- Mobile/desktop column-switching — original sniffs UA and adds `.single-column`. Here the responsive CSS in `styles.css` runs automatically off media queries.

For high-fidelity static mockups (slides, screenshots, single-page deliverables) this is sufficient. For a production rebuild, see `_research/index.html` for the original JS handlers.

## Styling

All visual styles live in `styles.css`, copied verbatim from the upstream `<style>` block. The class names match the source so you can paste markup directly. If you want token-driven semantic CSS instead, swap in `../../colors_and_type.css` and use the `.dp-*` utility classes from the parent design system.

## Adapting to new content

The component files use plain JS arrays (`PROJECTS`, `BOOKS`, `SOCIALS`, etc.) at the top. Edit those arrays to repurpose the kit for someone else's portfolio — the layout will hold as long as your data has the same shape (e.g. each project needs `emoji`, `name`, `desc`, `slug`, `href`).
