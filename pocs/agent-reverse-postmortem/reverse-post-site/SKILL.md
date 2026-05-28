---
name: reverse-post-site
description: Reads reverse-postmortem.md and renders a self-contained static site (reverse-postmortem-site/index.html) with a risk dashboard, per-incident cards, causal-chain visuals, and action-item checklists. No build step, no external dependencies.
allowed-tools: [Read, Glob, Bash, Write]
---

# Reverse Postmortem Site Renderer

You render a `reverse-postmortem.md` report into a clean, self-contained static site so the team can browse predicted incidents in the browser. You do not analyze code — you only transform the existing report.

## Global Context
- User request: $ARGUMENTS
- Input: `reverse-postmortem.md` (project root, or a path passed in `$ARGUMENTS`)
- Output: `reverse-postmortem-site/index.html`

## Rules
- Read-only on source. Write only inside `reverse-postmortem-site/`.
- If the report file does not exist, stop and tell the user to run `/reverse-post` first. Do not invent content.
- The site is a single self-contained `index.html` with inline CSS and JS. No external CDNs, no frameworks, no build step. It must open correctly via `file://`.
- Parse the report's structure (Risk Summary table, `## INC-*` sections, fields like Tier / Risk Score / Timeline / Root Cause / Detection Gap / Action Items). Preserve every incident and every action item.
- Do not change severity, scores, or wording of findings. You present; you do not re-judge.
- Tier colors: P0 red, P1 orange, P2 amber, P3 slate.

## Step 1 — Locate and read the report
- If `$ARGUMENTS` names a file, use it. Otherwise use `reverse-postmortem.md` in the project root.
- If missing, output: "No reverse-postmortem.md found. Run /reverse-post first." and stop.
- Read the full report.

## Step 2 — Parse into a model
Extract:
- Project name, generated date, tier counts from the header.
- The Risk Summary rows (incident #, title, likelihood, blast radius, risk, tier).
- For each `## INC-{n}` block: title, tier, risk score, predicted trigger, affected components, summary, timeline entries, root cause text + evidence file:line + code snippet, contributing factors, detection gap (existing signals / missing), blast radius detail, action items (table rows), earliest intervention point.
- Systemic patterns and appendix.

## Step 3 — Render reverse-postmortem-site/index.html

Create the directory and write a single self-contained file. Use this design:

- Dark theme, monospace-leaning, calm. Inline `<style>` and `<script>` only.
- **Header**: project name, generated date, and a big tagline: "These incidents have not happened yet."
- **Dashboard row**: four stat cards (P0/P1/P2/P3 counts) using tier colors; a total-incidents card.
- **Risk matrix**: a 5x5 grid (Likelihood x Blast Radius), each predicted incident plotted as a dot/badge in its cell, colored by tier. Axis labels. Pure CSS grid, no libs.
- **Filter bar**: buttons to filter incident cards by tier (All / P0 / P1 / P2 / P3). Toggle via inline JS.
- **Incident cards** (worst-first), each collapsible, with:
  - Title, tier badge (colored), `L x B = risk` chip.
  - Predicted trigger and affected components.
  - Summary.
  - A causal-chain strip rendered as boxes with arrows: Trigger -> Fragile Code -> Propagation -> Impact (derive the four nodes from the timeline + root cause).
  - Timeline as a vertical list.
  - Root cause with the code snippet in a `<pre>` block and the `file:line` shown as a label.
  - Detection gap shown as a callout (red if "none" existing signals).
  - Action items as a checklist (`<input type=checkbox>`), each with effort badge (S/M/L) and "prevents" note. Checkbox state is client-side only.
  - Earliest intervention point highlighted in a distinct box.
- **Systemic patterns** section near the bottom.
- **Footer**: note that incidents are predictions, not history; link back to the source `reverse-postmortem.md`.

Keep the JS minimal: tier filtering, card collapse/expand, checkbox toggling. No external requests.

Escape any HTML special characters in code snippets so they render literally.

## Step 4 — Report back
- Print the path to `reverse-postmortem-site/index.html`.
- Print how many incidents were rendered and the tier breakdown.
- Suggest opening it: `open reverse-postmortem-site/index.html` (macOS) or the platform equivalent.
