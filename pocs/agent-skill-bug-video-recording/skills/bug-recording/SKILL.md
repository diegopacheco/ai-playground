---
name: bug-recording
description: Detect whether a target web project is a React app on Node or Bun, run it, hunt for CSS / functional / render bugs with Playwright, and record an optimized video of each bug. Use when the user runs /bug-recording or asks to find and record UI bugs in an app.
---

# bug-recording

Find bugs in a running React app and record a Playwright video of each one.

## Inputs

- Target app directory. Default: current project root.
- Output directory. Default: `<target>/bug-recording-output`.

## Preconditions

Verify these are on PATH and stop loudly if any is missing:

- `node` (or `bun`) to run the target app.
- `ffmpeg` for video encoding and the size-reduction pass.
- This skill's own dependencies. If `node_modules/playwright` is absent in the
  skill directory, run `npm install` here, then `npx playwright install chromium`.

## Run

From the skill directory:

```
node record.mjs <target-app-dir> [output-dir]
```

The script:

1. Detects the stack from `package.json` and lockfiles (React + Node/Bun, the
   package manager, the dev script).
2. Starts the dev server and waits until it reports a URL and is reachable.
3. Discovers routes by crawling in-app links from the root page.
4. Per route, captures console errors and runs three detectors:
   - render: a console/page error on load or an error-boundary fallback,
   - css: text clipped under `overflow:hidden` with no ellipsis,
   - functional: an "increase"/"+"/"add" control that moves a numeric readout
     the wrong way.
5. Records a narrated Playwright walkthrough per confirmed bug: starts on the
   home page, moves a visible cursor to the matching nav link and clicks through
   to the bug page, captions each reproduction step on screen, performs the
   triggering click, and dwells on the result with the broken element
   highlighted. Then converts it to H.264 `.mp4`, runs `reduce-size.sh` to shrink
   it, and samples keyframes.
6. Writes `findings.json` plus `videos/`, `frames/`, `screenshots/`.

## After running

- Report the bug count and where artifacts landed.
- The bugs, evidence, and per-step action logs are in `findings.json`. You may
  read the sampled `frames/*.png` to enrich each bug's reproduction steps before
  handing off to `/bug-report`.

## Honest limits

The detectors target broad bug classes, not every possible bug. Report the
captured evidence, not a verdict beyond it. The agent cannot watch the `.mp4` as
motion; reproduction narrative comes from the action log and sampled frames.
