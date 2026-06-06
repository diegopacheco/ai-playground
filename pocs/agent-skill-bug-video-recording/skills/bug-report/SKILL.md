---
name: bug-report
description: Render a self-contained light-theme website with three tabs (bugs found, bug videos, reproduction steps) from a bug-recording output directory. Use when the user runs /bug-report or asks to turn recorded bug findings into a report site.
---

# bug-report

Turn a `/bug-recording` output directory into a static report site.

## Inputs

- Findings directory (the `/bug-recording` output that holds `findings.json`,
  `videos/`, and `screenshots/`). Default: `./bug-recording-output`.
- Site output directory. Default: `<findings>/../bug-report-site`.

## Preconditions

- `findings.json` must exist in the findings directory. If it does not, run
  `/bug-recording` first and stop loudly.

## Run

```
node report.mjs <findings-dir> [site-dir] [--serve] [--port=7800]
```

The script reads `findings.json`, copies each `.mp4` and screenshot into the
site's `assets/`, and writes `index.html`. With `--serve` it then serves the
site over HTTP (with byte-range support so videos seek) and prints the URL.

## The site

A single self-contained HTML file (inline CSS and JS, light theme) with three
tabs:

1. **Bugs Found** — a card per bug: class badge, description, route, component,
   and a one-line evidence summary.
2. **Videos** — the recorded `.mp4` per bug, selectable from a list, each using
   the captured screenshot as its poster.
3. **Reproduction Steps** — the ordered `navigate` / `click` / `observe` steps
   per bug, reconstructed from the captured action log and sampled frames.

## After running

- Report the site path and open it. The report is a standalone static site — it
  is **not** served by the target app's dev server, so do not load it from the
  app's port (that returns 404). Either open the file directly:

  ```
  open bug-report-site/index.html
  ```

  or serve it on its own port and open the printed URL:

  ```
  node report.mjs <findings-dir> --serve --port=7800
  ```

- You may enrich each bug's reproduction steps in `findings.json` (from reading
  the sampled `frames/*.png`) before regenerating the site.
