# Agentic Project Logo Generation

This document describes how the 177 catalog logos are selected, generated, normalized, named, validated, and published.

## Result

- 177 PNG files
- 528 × 528 pixels per file
- White or transparent visual background
- Deterministic filename based on catalog ID and project name
- Existing project-owned logo preferred when available
- Generated logo used only when the project has no owned mark
- Lazy-loaded by the landing page

All final files live in `agents/logos`.

## Technology

- macOS on Apple Silicon
- Draw Things CLI `1.20260430.0`
- Local `flux_1_schnell_q5p.ckpt` model
- Node.js built-in modules
- macOS `sips` for resizing and padding
- macOS Quick Look for SVG rasterization
- `curl` for owned logos stored in separate GitHub repositories
- GitHub Pages for publishing

No hosted image API or paid image service is used.

## Main Files

### `agents/generate-logos.mjs`

The batch runner performs the full asset workflow:

1. Reads the 177 projects from `agents/landing-page.html`.
2. Builds a deterministic filename from the project ID and name.
3. Checks the owned-logo map.
4. Downloads or reads an owned logo when one exists.
5. Rasterizes SVG sources when required.
6. Resizes owned marks to a maximum of 480 pixels.
7. Pads owned marks to a 528 × 528 white canvas.
8. Generates a logo with Draw Things when no owned mark exists.
9. Uses a deterministic seed equal to `1000 + project ID`.
10. Resizes generated output to exactly 528 × 528.
11. Skips existing generated files so interrupted runs can resume.

### `agents/landing-page.html`

The catalog derives each logo path from the same ID and slug rules used by the batch runner. Every image has fixed width and height attributes, lazy loading, and asynchronous decoding. Archive cards use rendering containment so off-screen content does not delay the first render.

### `.github/workflows/agents-pages.yml`

The publishing workflow copies both the landing page and `agents/logos` into the GitHub Pages artifact.

## Owned Logo Policy

The batch runner contains an explicit map of project-owned logo sources. Sources can be:

- A file under `pocs`
- A file in a related local repository
- A raw file from a separate GitHub repository

Framework boilerplate marks are excluded. Project-owned marks replace generated marks and are normalized to the same 528 × 528 output contract.

The current catalog contains 17 owned marks and 160 locally generated marks.

## Generation Prompt

Each generated logo prompt includes:

- Project name
- Project description
- Category-specific visual motif
- Deterministic rotating color palette
- Centered single-symbol composition
- Flat geometric treatment
- White background requirement
- No text, letters, watermark, border, shadow, or gradient

The negative prompt rejects typography, watermarks, frames, photographs, shadows, gradients, clutter, busy backgrounds, and multiple marks.

## Generation Settings

The completed batch used Flux Schnell with fast local settings for most assets:

```text
Model: flux_1_schnell_q5p.ckpt
Steps: 2
CFG: 1
Native size: 384 × 384
Final size: 528 × 528
Seed: 1000 + project ID
Output: PNG
```

Some initial assets were produced at 512 × 512 with four steps before the faster settings were validated. Every final file uses the same 528 × 528 contract.

## Install the CLI

```bash
brew install drawthingsai/draw-things/draw-things-cli
draw-things-cli models list --downloaded-only
```

Draw Things resolves models from:

```text
~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models
```

## Run the Complete Batch

```bash
node agents/generate-logos.mjs --steps=2 --size=384
```

## Run a Range

```bash
node agents/generate-logos.mjs --from=120 --to=140 --steps=2 --size=384
```

## Replace Existing Generated Files

```bash
node agents/generate-logos.mjs --from=120 --to=140 --steps=2 --size=384 --force
```

Owned marks are always refreshed from their declared source. Generated marks are replaced only when `--force` is present.

## Use Another Installed Model

```bash
node agents/generate-logos.mjs --model=pixelwave_flux_1_schnell_04_q8p.ckpt --steps=4 --size=512
```

The native size must be divisible by 64 because Draw Things enforces that constraint. The final resize remains 528 × 528.

## Validation

Count final PNG files:

```bash
find agents/logos -maxdepth 1 -type f -name '*.png' | wc -l
```

Check dimensions:

```bash
find agents/logos -maxdepth 1 -type f -name '*.png' -exec sips -g pixelWidth -g pixelHeight {} \;
```

Check temporary files:

```bash
find agents/logos -maxdepth 1 -type f -name '.*'
```

The expected result is 177 PNG files, every file at 528 × 528, with no temporary files.

## Performance

The landing page avoids loading all 177 images at startup:

- `loading="lazy"`
- `decoding="async"`
- Fixed 528 × 528 dimensions
- `content-visibility: auto` on archive cards
- `contain-intrinsic-size` for stable off-screen layout
- One reusable image per project across highlighted and archive cards

The browser cache reuses a logo when the same project appears in both sections.
