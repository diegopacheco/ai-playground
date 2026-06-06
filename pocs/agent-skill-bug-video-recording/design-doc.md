# Design Doc — `/bug-recording` & `/bug-report`

Status: draft for review
Owner: diegopacheco
Date: 2026-06-05

## 1. Summary

Two Claude skills plus a sample app and a generated report site.

- **`/bug-recording`** — points at a web project, figures out whether it is a
  React app running on Node or Bun, starts it, drives it with Playwright, hunts
  for bugs using runtime heuristics, and records a Playwright video of each bug
  it reproduces. Output is a set of artifacts: `findings.json`, one optimized
  `.mp4` per bug, sampled frames, and per-step screenshots/logs.
- **`/bug-report`** — reads those artifacts and renders a self-contained static
  website with three tabs: the bug list, the bug videos, and reproduction steps.

A sample app is shipped so both skills have a real target with known, planted
bugs: at least one CSS bug, one functional bug, one render bug.

`install.sh` / `uninstall.sh` install and remove both skills from the global
Claude skills directory (`~/.claude/skills`).

## 2. Goals / Non-Goals

### Goals
- Detect stack: React + (Node | Bun) from project files, with a clear fallback.
- Start the target app and wait for it to be reachable.
- Find bugs of three broad classes (CSS / functional / render) by runtime signal.
- Record one optimized video per bug with Playwright, post-processed by ffmpeg.
- Produce machine-readable findings and a human-facing 3-tab site.
- One-command install / uninstall into global Claude.

### Non-Goals
- General-purpose bug finding for arbitrary apps. The heuristics target broad
  classes; the sample app's bugs are chosen to trip them. We state this honestly.
- The agent literally watching motion video. It cannot. Repro steps and analysis
  come from the captured action log plus ffmpeg-sampled frames read as images.
- Fixing the bugs. The skills find, record, and report. They do not patch.
- Framework coverage beyond React/Vite for the sample app.

## 3. Key Decisions & Assumptions

1. **Two skills, shared artifact directory.** `/bug-recording` writes to
   `bug-recording-output/` in the target project; `/bug-report` reads it. Cleaner
   than one skill with modes; either can run alone given the artifacts.
2. **Playwright library, not the Playwright MCP, for video.** The MCP tools only
   expose stills. True session video needs `recordVideo` on a browser context.
3. **ffmpeg is a hard dependency**, on PATH. Raw Playwright `.webm` is VFR/VP8;
   we re-encode to H.264 `.mp4` so `<video>` plays it everywhere in the report.
4. **The agent cannot watch video.** Tab 3 repro steps are reconstructed from the
   instrumented action log and ffmpeg-sampled keyframes (read as PNGs), not from
   playing the file. The doc and the report are explicit about this.
5. **Sample app stack:** Vite + React 19 + TypeScript 6 + TanStack Router +
   TanStack Query, modular feature folders, runnable under both Node and Bun so
   stack detection has something real to detect. (TanStack Start considered and
   rejected — a meta-framework complicates the "node vs bun runner" signal.)
6. **TypeScript 6** as requested; no downgrade.
7. **Report site is self-contained static HTML** generated into
   `bug-report-site/`. Bespoke, polished, **light theme** UI (frontend-design
   approach), not a reuse of the existing `*-site` look. Videos are copied into
   the site so it is portable.
8. **Bounded, deterministic crawl.** The recorder discovers routes from the
   TanStack route tree (or DOM links) and interacts with a bounded set of
   elements, re-checking signals after each step. No unbounded fuzzing.

## 4. Architecture Overview

Pipeline, left to right:

- **Target project** → **Stack Detector** reads `package.json`, lockfiles, and
  engines to classify React + Node/Bun and pick the run command.
- **App Runner** installs deps and starts the dev server, waiting on the port.
- **Bug Hunter** (Playwright) crawls routes, captures console / network /
  screenshot / DOM snapshot per step, and runs the three heuristic checks.
- **Recorder** (Playwright `recordVideo`) replays the steps that triggered each
  confirmed bug into a focused `.webm`.
- **ffmpeg** re-encodes each `.webm` → `.mp4` and samples keyframes → `.png`.
- **Findings Writer** emits `findings.json` plus the artifact tree.
- **`/bug-report`** reads `findings.json` and renders `bug-report-site/`.

The README renders this as a hand-drawn (Excalidraw-style) SVG diagram — wobble
filter, Caveat handwriting font, pastel boxes, solid capture path. No ASCII.

## 5. Component Design

### 5.1 `/bug-recording` skill

`SKILL.md` instructs Claude to run the pipeline. Stages:

**A. Detect stack.** Signals, in priority order:
- `bun.lockb` or `bun` in `packageManager`/`engines` → Bun runner.
- `package-lock.json` / `pnpm-lock.yaml` / `yarn.lock` → Node runner (npm/pnpm/yarn).
- `react` + `react-dom` in dependencies → React app.
- Dev command from `scripts.dev` (fallback `scripts.start`).
- Port discovered from Vite output / config (fallback scan of common ports).
If React is not detected, the skill stops loudly and reports why.

**B. Run the app.** Install deps with the detected manager, start the dev server
in the background, and wait for the port with a bounded poll loop (sleep 1).

**C. Hunt for bugs.** With Playwright (headed context for recording fidelity):
- Enumerate routes from the route tree or by crawling in-app links.
- Per route: capture console messages, network requests, a screenshot, and the
  accessibility/DOM snapshot.
- Interact with a bounded set of controls (buttons, steppers, inputs, filters),
  re-capturing signals after each interaction.
- Apply the three detectors (section 7). Each positive becomes a candidate bug
  with: class, page/route, component guess, triggering action sequence, evidence.

**D. Record.** For each confirmed bug, open a fresh context with
`recordVideo`, replay only the triggering action sequence, then `context.close()`
to flush the `.webm`.

**E. Optimize.** ffmpeg `.webm` → `.mp4` (H.264, yuv420p, even dims), and sample
keyframes to `frames/`.

**F. Emit** `findings.json` and the artifact tree.

### 5.2 Sample app

`sample-app/` — Vite + React 19 + TS 6 + TanStack Router + TanStack Query.
Modular structure: `src/features/<feature>/{components,hooks,api,routes}`.
A small in-memory product/cart surface gives the bugs somewhere to live.
Runs under Node (`npm run dev`) and Bun (`bun run dev`).

Planted bugs (each maps to one detector so the design is verifiable):

| # | Class      | Where                         | The bug                                                              | Detection signal |
|---|------------|-------------------------------|---------------------------------------------------------------------|------------------|
| 1 | Render     | `features/dashboard/Stats`    | Maps over a field that can be `undefined`, throws on a route visit   | React error in console + error-boundary fallback + missing node |
| 2 | Functional | `features/cart/QtyStepper`    | "+" handler decrements (sign bug); cart count moves the wrong way    | Action-vs-effect: click "+", number decreases |
| 3 | CSS        | `features/catalog/ProductCard`| Long title in an `overflow:hidden` box, no ellipsis → clipped/overflow| `scrollWidth > clientWidth`, content overflows container box |

### 5.3 `/bug-report` skill

`SKILL.md` instructs Claude to read `findings.json` + artifacts and generate
`bug-report-site/index.html` (self-contained: inline CSS/JS, videos copied to
`bug-report-site/assets/`). Three tabs:

- **Tab 1 — Bugs found.** Card per bug: class badge, description, page/route,
  component name, evidence summary.
- **Tab 2 — Video.** `<video controls>` per bug playing the `.mp4`, selectable
  from the bug list.
- **Tab 3 — Repro steps.** Ordered steps the agent reconstructed from the action
  log + sampled frames, with a note that steps derive from captured actions/frames.

UI: bespoke, polished, **light theme** (light surfaces, soft shadows, pastel
accents), production-grade. No dark mode.

## 6. Data Contract — `findings.json`

Shape (contract, not implementation):

```
{
  "target": { "path", "stack": { "framework": "react", "runtime": "node|bun", "runCommand", "port" } },
  "generatedAt": "ISO-8601",
  "bugs": [
    {
      "id": "bug-1",
      "class": "render | functional | css",
      "title": "short description",
      "page": "/route",
      "component": "Stats",
      "evidence": { "console": [...], "network": [...], "metrics": {...} },
      "steps": [ { "n": 1, "action": "navigate", "target": "/dashboard" }, ... ],
      "video": "videos/bug-1.mp4",
      "frames": ["frames/bug-1-001.png", ...],
      "screenshot": "screenshots/bug-1.png"
    }
  ]
}
```

`/bug-report` depends only on this contract, so the two skills stay decoupled.

## 7. Bug Detection Strategy (and its limits)

- **Render** — any React error / hydration warning in the console, an error
  boundary fallback in the DOM, or an expected region rendering empty/blank.
- **Functional** — an interactive control whose observed effect contradicts its
  intent (label/aria), or a `4xx/5xx` on an action's network call. The stepper
  bug is detected by asserting the displayed number moves opposite the control.
- **CSS** — layout faults measured structurally: bounding-box overlap of
  interactive elements, content overflowing its container, clipped text
  (`scrollWidth > clientWidth` under `overflow:hidden` with no ellipsis), or
  elements outside the viewport.

Honest limits (Rule 12): these catch broad classes, not every bug. Frame
sampling can miss timing-class faults (flicker, animation that never settles).
For arbitrary apps this is best-effort; the sample app's bugs are chosen to be
caught. The report never claims more certainty than the evidence supports.

## 8. Video Pipeline

1. Playwright context with `recordVideo: { dir, size: 1280x720 }`.
2. Replay the bug's triggering steps; `context.close()` flushes the `.webm`.
3. ffmpeg → `.mp4`: `libx264`, `-crf 28`, `-pix_fmt yuv420p`, even dimensions via
   `scale=trunc(iw/2)*2:trunc(ih/2)*2`, normalized frame rate.
4. **Size-reduction pass (required).** After the `.mp4` is produced, a dedicated
   `reduce-size.sh` ffmpeg script shrinks it for web embedding: higher CRF
   (`-crf 32`), `-preset veryslow`, `-movflags +faststart`, capped width
   (`scale='min(1280,iw)':-2`), and audio dropped (`-an`). It only replaces the
   original if the result is actually smaller, and logs before/after bytes.
5. ffmpeg keyframe sample (`fps=1` plus scene-change frames) → `frames/` for the
   agent to read as images when reconstructing repro steps.

## 9. install.sh / uninstall.sh

- `install.sh`: verify Node or Bun, Playwright, and ffmpeg are present (fail loud
  if not); copy `skills/bug-recording` and `skills/bug-report` into
  `~/.claude/skills/`; print the installed paths. No comments, no emojis, any wait
  uses a bounded loop with sleep 1.
- `uninstall.sh`: remove both skill directories from `~/.claude/skills/`; print
  what was removed. No comments, no emojis.

## 10. README & Diagram Plan

- Hand-drawn (Excalidraw-style) architecture SVG: wobble (`feTurbulence` +
  `feDisplacementMap`), Caveat font, pastel node fills, solid arrows along the
  capture path. Nodes: Target → Detect → Run → Hunt → Record → ffmpeg → Findings
  → Report (3 tabs). No ASCII art.
- `printscreens/` holds Playwright-captured screenshots of: the running sample
  app, each bug, and all three report tabs. README loads and explains each.

## 11. Directory Layout

```
agent-skill-bug-video-recording/
  design-doc.md
  README.md
  install.sh
  uninstall.sh
  skills/
    bug-recording/SKILL.md   (+ helper scripts)
    bug-report/SKILL.md      (+ site template/generator)
  sample-app/                (Vite + React 19 + TS6 + TanStack)
  printscreens/
```

`/bug-recording` writes `bug-recording-output/` into its target; `/bug-report`
emits `bug-report-site/`. Both are generated, not committed sources.

## 12. Dependencies

- Runtime: Node 24 or Bun 1.3 (both present locally).
- Playwright (library + Chromium) and ffmpeg 8 (present locally).
- Sample app: react 19, react-dom 19, @tanstack/react-router,
  @tanstack/react-query, vite, typescript 6. Kept minimal.

## 13. Risks & Open Questions

Risks:
- Crawl/interaction may be brittle on dynamic apps → keep it bounded and
  route-tree driven; degrade to link-crawl.
- Heuristics yield false positives/negatives → report evidence, not verdicts.
- Video size/codec portability → ffmpeg H.264 mp4 settles it.

Resolved:
1. TanStack scope — **Router + Query**.
2. **Two skills** (`/bug-recording`, `/bug-report`) sharing the artifact dir.
3. Report styling — **bespoke, polished, light theme**.
4. Video — **dedicated ffmpeg size-reduction pass** after the mp4 encode.

Still defaulted (flag if wrong):
- Recorder targets any path the user passes; the bundled sample app is the
  default known-good target.

## 14. Build Order

1. Sample app with the 3 planted bugs (verifiable target).
2. `/bug-recording`: detect → run → hunt → record → ffmpeg → `findings.json`.
3. `/bug-report`: 3-tab static site from `findings.json`.
4. `install.sh` / `uninstall.sh`.
5. README with hand-drawn SVG diagram + `printscreens/`.
6. End-to-end pass on the sample app; capture screenshots.
