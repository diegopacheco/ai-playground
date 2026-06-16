---
name: accessibility-auditor
description: Audits a running web page for WCAG accessibility issues, captures a screenshot and annotates it with a numbered box over every problem, then renders a self-contained light-theme website with the score, the annotated screenshot, and a fix for each finding. Use when the user runs /accessibility-auditor or asks to audit, check, or report a site's accessibility, WCAG compliance, alt text, color contrast, labels, or screen-reader support.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# Accessibility Auditor

When invoked, you load a real web page in a headless Chromium, run a set of WCAG 2.1 AA checks against the live DOM and computed styles, take a full-page screenshot, and produce a light-theme report that overlays a numbered box on the screenshot for every issue and lists the fix for each. Every finding is read from the running page — nothing is estimated.

## Global Context
- User request / scope: $ARGUMENTS — empty, a URL, or a path to a web project
- Engine: `scripts/audit.mjs` (Node; its only dependency is puppeteer, installed beside the skill with its own Chromium)
- Template: `assets/template.html`
- Output: `a11y-report/index.html`, `a11y-report/data.json`, and `a11y-report/screenshot.png` in the current directory

## What it checks
Images without `alt` (1.1.1), form controls with no label (4.1.2), buttons and links with no accessible name (4.1.2), text below the AA contrast minimum (1.4.3), heading levels that skip (1.3.1), missing `<main>` landmark (1.3.1), missing page `lang` (3.1.1), empty document title (2.4.2), missing `<h1>` (2.4.6), positive `tabindex` (2.4.3), `role="button"` that is not keyboard focusable (2.1.1), vague link text (2.4.4), and duplicate ids (4.1.1).

## Rules
- Every issue is detected from the live DOM and real computed styles, never guessed. Page-level issues that have no on-screen box are listed without a marker.
- The engine only writes inside `a11y-report/`. It does not modify the audited project.
- Do not add comments to any command you run.

## Step 1 — Decide the target URL
Read `$ARGUMENTS`:
- A URL (starts with `http`) → audit it directly, skip to Step 3.
- A path to a project, or empty → treat it as a web project to start locally (Step 2).

## Step 2 — Start the project if you only have a path
If you do not already have a running URL, start the project's dev server in the background and wait for it to answer:

```bash
cd <project> && (npm run dev >/tmp/a11y-dev.log 2>&1 &) ; for i in $(seq 1 60); do curl -sf http://localhost:5188 >/dev/null && break; sleep 1; done
```

Read the project's `vite.config`/`package.json` for the real port if it is not 5188. If dependencies are missing, run `npm --prefix <project> install` first.

## Step 3 — Run the audit engine
Invoke the engine by its installed absolute path. It writes `a11y-report/` into the current working directory:

```bash
node "$HOME/.claude/skills/accessibility-auditor/scripts/audit.mjs" <url> a11y-report
```

The engine loads the page, runs the checks, scores the page 0-100 (A–F), captures the screenshot, computes each issue's box as a percentage of the image, and writes the report. It prints a summary to stdout. If the page cannot be loaded it says so — relay that instead of producing an empty report.

## Step 4 — Report back
Relay the stdout summary in your own words: the score and grade, how many issues by severity, and the worst offenders by WCAG criterion. Name the highest-impact fixes (missing alt text, unlabeled inputs, low contrast). Point the user at the site:

```
a11y-report/index.html
```

Offer to open it. The page is self-contained — a score gauge and grade, severity cards, the annotated screenshot with a clickable numbered box per issue, a by-principle breakdown (Perceivable / Operable / Understandable / Robust), and a filterable card per finding with its WCAG criterion, the element selector and snippet, and the fix. No server or network needed.

## What the numbers mean
- **Score** — starts at 100; each issue deducts by impact (critical 15, serious 10, moderate 5, minor 2), floored at 0. Grade A ≥90, B ≥75, C ≥60, D ≥40, else F.
- **Impact** — critical (blocks a screen-reader user, e.g. no alt / no label), serious (contrast, keyboard reachability, page language), moderate (structure: headings, landmarks, focus order), minor (link wording, duplicate ids).
- **Box** — stored as a percentage of the captured image so the overlay lines up at any display size. Page-level issues (lang, title, missing main) have no box and are listed without a marker.
