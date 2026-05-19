# Fonts

The portfolio uses **Inter** (weights 400 / 500 / 600 / 700), loaded from Google Fonts via:

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
```

This is the canonical reference and is already declared at the top of `colors_and_type.css`.

## Self-hosting

If you need to ship offline / sandboxed:

1. Download from https://fonts.google.com/specimen/Inter — pick "Variable" or the 4 static weights.
2. Place the WOFF2 files in this folder.
3. Replace the `@import` in `colors_and_type.css` with `@font-face` declarations.

> ⚠️ **No font files are bundled in this design system.** The site has always relied on the Google Fonts CDN. Please confirm with the brand owner whether self-hosting is required for any new deliverable.

## Substitution

If Inter is unavailable for any reason, the closest Google Fonts matches are:
- **DM Sans** — slightly warmer, otherwise nearly identical proportions
- **IBM Plex Sans** — slightly more humanist, same weight range
- System fallback: `ui-sans-serif, system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif` (already declared in `--font-sans`).
