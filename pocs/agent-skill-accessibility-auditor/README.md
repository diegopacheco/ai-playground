# accessibility-auditor — WCAG audit skill

An agent skill that answers one question about any web page: **what accessibility problems does it have, and where are they on screen?** It loads the page in a real headless Chromium, runs WCAG 2.1 AA checks against the **live DOM and computed styles**, captures a full-page screenshot, and renders a self-contained, light-theme website that draws a **numbered box on the screenshot for every issue** and lists the fix for each.

> The task: *"Build a Claude skill with install/uninstall scripts for global Claude, a `sample/` app (Node.js, React 19, Vite, TanStack) that has accessibility issues, and a skill — called as `/accessibility-auditor` — that annotates a screenshot with WCAG issues, does all the accessibility analysis, and produces a light-theme website with the results."*

## How it works

![Architecture](printscreens/architecture.png)

1. **Target page** — a URL, or a local project whose dev server the skill starts.
2. **Headless Chromium** — `puppeteer` (its only dependency, with its own bundled Chromium) loads the page.
3. **WCAG checks** — a script runs inside the page against the real DOM and computed styles, so contrast, labels, and roles are measured, not guessed.
4. **Full-page screenshot** — captured at a fixed width; each issue's bounding box is recorded.
5. **Score + annotate** — the page is scored 0–100 (A–F) and each box is stored as a percentage of the image, so the overlay lines up at any display size.
6. **Light report** — the data and the screenshot are injected into an HTML template written to `a11y-report/index.html`.

Every finding is read from the running page. Page-level issues that have no on-screen box (missing `lang`, empty `<title>`, no `<main>`) are listed without a marker.

## What it checks

| WCAG | Criterion | Impact | What it flags |
|---|---|---|---|
| 1.1.1 | Non-text Content | critical | `<img>` with no `alt` |
| 4.1.2 | Name, Role, Value | critical | inputs with no label; buttons/links with no accessible name |
| 1.4.3 | Contrast (Minimum) | serious | text below the 4.5:1 (or 3:1 for large text) ratio |
| 3.1.1 | Language of Page | serious | `<html>` with no `lang` |
| 2.1.1 | Keyboard | serious | `role="button"` that is not keyboard focusable |
| 1.3.1 | Info and Relationships | moderate | heading levels that skip; no `<main>` landmark |
| 2.4.2 / 2.4.6 | Page Titled / Headings | moderate | empty `<title>`; no `<h1>` |
| 2.4.3 | Focus Order | moderate | positive `tabindex` |
| 2.4.4 | Link Purpose | minor | vague link text ("read more", "click here") |
| 4.1.1 | Parsing | minor | duplicate `id` |

## The report

A score gauge and grade, severity cards, and an issues-by-principle breakdown (Perceivable / Operable / Understandable / Robust):

![Report top](printscreens/report-top.png)

The centerpiece — the captured screenshot with a clickable numbered box over every issue, colored by impact. Clicking a box jumps to its finding:

![Annotated screenshot](printscreens/report-annotated.png)

The full page, including the filterable findings list where each card carries the WCAG criterion, the element selector and HTML snippet, the measured contrast colors, and the concrete fix:

![Report full](printscreens/report-full.png)

## Try it on the sample app

A real Vite + **React 19** + **TanStack Router** storefront lives in `sample/` — *Pixel Pantry*. It looks fine, which is the point: the accessibility problems are invisible to a sighted user but real to a screen reader.

![Sample app](printscreens/sample-app.png)

It carries these issues on purpose:

| Issue | WCAG | Where |
|---|---|---|
| 6 product images with no `alt` | 1.1.1 | product grid |
| search box and email field with no label | 4.1.2 | hero, newsletter |
| cart icon button with no accessible name | 4.1.2 | header |
| filter chips are `<div role="button">`, not focusable | 2.1.1 | hero |
| gray subtitle, "save 20%", and footer text below contrast | 1.4.3 | throughout |
| `<html>` has no `lang` | 3.1.1 | `index.html` |
| no `<main>` landmark | 1.3.1 | layout |
| heading jumps `h1` → `h3` | 1.3.1 | hero |
| positive `tabindex` on the email field | 2.4.3 | newsletter |
| `id="title"` used twice | 4.1.1 | hero + newsletter |
| "Read more" link with no context | 2.4.4 | footer |

Run the whole pipeline — installs deps, starts the app, audits it, writes the report:

```bash
./test.sh
```

Measured result (`test.sh` output):

```
accessibility audit  http://localhost:5188/
score 0/100  grade F  issues 29
  critical 8  serious 16  moderate 3  minor 2
PASS report written to a11y-report/
```

29 findings across 9 WCAG criteria; the score floors at 0 because eight critical issues already exceed the deduction budget. Open `a11y-report/index.html` to see them on the screenshot.

### Run the live sample

```bash
./sample/start.sh      # vite dev on http://localhost:5188
./sample/stop.sh
```

## Install

```bash
./install.sh
```

Copies the skill to `~/.claude/skills/accessibility-auditor` (and `~/.codex/skills/accessibility-auditor` if Codex is present) and runs `npm install` there, which downloads puppeteer and its Chromium. Requires `node` and `npm`.

## Uninstall

```bash
./uninstall.sh
```

## Usage

In Claude Code, point the skill at a running URL:

```
/accessibility-auditor http://localhost:5188
```

Or at a web project — it starts the dev server for you:

```
/accessibility-auditor ./sample
```

It writes `a11y-report/index.html`, `a11y-report/data.json`, and `a11y-report/screenshot.png` into the current directory and prints a summary.

## What the numbers mean

- **Score** — starts at 100; each issue deducts by impact (critical 15, serious 10, moderate 5, minor 2), floored at 0. Grade A ≥90, B ≥75, C ≥60, D ≥40, else F.
- **Impact** — critical (blocks a screen-reader user: no alt, no label), serious (contrast, keyboard reachability, page language), moderate (structure: headings, landmarks, focus order), minor (link wording, duplicate ids).
- **Box** — stored as a percentage of the captured image so the overlay stays aligned at any size.

## Layout

```
agent-skill-accessibility-auditor/
├── skill/
│   ├── SKILL.md                 the agent playbook
│   ├── package.json             puppeteer
│   ├── scripts/audit.mjs        load + WCAG checks + screenshot + score + render
│   └── assets/template.html     the light-theme report template
├── sample/                      Vite + React 19 + TanStack Router storefront
│   ├── start.sh / stop.sh
│   └── src/routes/              root layout + home + about, issues on purpose
├── printscreens/                architecture diagram + screenshots
├── install.sh / uninstall.sh
├── test.sh                      end-to-end: start sample, audit, write report
└── README.md
```
