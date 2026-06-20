---
name: habit-tracker
description: Reads every Claude Code session in the user's config directory and renders a self-contained light-theme website with a GitHub-style contribution grid, visual metrics, and 5-10 written insights about the user's agent coding habits and what they are building. Use when the user runs /habit-tracker or asks about their coding habits, activity streak, contribution graph, what they have been building, or a recap of their Claude Code usage.
allowed-tools: [Bash, Read]
---

# Coding Habit Tracker

When invoked, you read every past Claude Code session stored under the user's config directory, aggregate the activity, and produce a light-theme report with a GitHub-style contribution grid, summary metrics, charts, and a set of written insights about how and when the user codes and what they are building. Every number is read from the session logs — nothing is estimated.

## Global Context
- User request / scope: $ARGUMENTS — usually empty
- Engine: `scripts/analyze.mjs` (Node, no dependencies)
- Template: `assets/template.html`
- Source data: `$CLAUDE_CONFIG_DIR/projects/**/*.jsonl` (defaults to `~/.claude/projects`)
- Output: `habit-report/index.html` and `habit-report/data.json` in the current directory

## What it reads
Each session line carries a timestamp, the working directory, the git branch, the model, and the message content. The engine counts user prompts and assistant turns per day, hour, and weekday; every `tool_use` by tool name; sessions and their duration; projects by working-directory name and their git branches; and classifies prompts as building, fixing, refactoring, testing, or exploring from their wording.

## What it produces
- A contribution grid of the last year, one square per day, shaded by interaction volume.
- Stat cards: sessions, active days, current streak, longest streak, tool calls, projects.
- Charts: most-used tools, hour-of-day, day-of-week, and top projects.
- 5-10 written insights chosen from the most salient signals for this user.

## Rules
- Every figure is derived from the real session logs, never guessed.
- The engine only writes inside `habit-report/`. It never modifies the session logs or any project.
- Do not add comments to any command you run.

## Step 1 — Run the engine
Invoke it by its installed absolute path. It writes `habit-report/` into the current working directory:

```bash
node "$HOME/.claude/skills/habit-tracker/scripts/analyze.mjs" habit-report
```

It scans the session files, computes the aggregates, picks the insights, and writes the report. It prints a summary to stdout. If no session files are found it says so — relay that instead of producing an empty report.

## Step 2 — Report back
Relay the stdout summary in your own words: total interactions and sessions, active days and the date span, the current and longest streak, the busiest hour and weekday, and the top project. Read a couple of the generated insights from `habit-report/data.json` and mention them. Point the user at the site:

```
habit-report/index.html
```

Offer to open it. The page is self-contained — the grid, cards, charts, and insight cards are all inline, with no server or network needed.

## What the numbers mean
- **Interaction** — one user prompt or one assistant turn in a session.
- **Active day** — a calendar day (local time) with at least one interaction.
- **Current streak** — consecutive days up to today that each have activity. **Longest streak** — the longest such run ever.
- **Grid level** — each day's square is shaded in five steps by its interaction count relative to the busiest day of the year.
- **Insight priority** — every candidate insight carries a salience score; the engine renders the top ten that apply to this user's data, so the set changes as habits change.
