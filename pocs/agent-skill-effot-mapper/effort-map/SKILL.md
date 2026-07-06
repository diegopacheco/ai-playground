---
name: effort-map
description: Read the current project's current and prior Claude Code or Codex sessions, extract every prompt, score five effort metrics 0-100 with reasons, estimate average tokens and time per session, and render a self-contained light-theme HTML report. Use when the user runs /effort-map or asks how much effort, thought, bug-fixing, UX tuning, architecture care, or copy-paste slop went into a project.
---

# Effort Map

Score how much real effort went into the current project by reading its agent sessions, then produce `effort-DD-MM-YYYY-report.html`.

## Workflow

1. Detect the active agent: `claude` when running in Claude Code, `codex` when running in Codex.
2. Collect the sessions for the current project. From this skill directory run:
   ```
   python3 scripts/collect_sessions.py --project "$PWD" \
     --output /tmp/effort-map-sessions.jsonl --stats /tmp/effort-map-stats.json
   ```
   Add `--agent claude` or `--agent codex` only if automatic detection fails. The command prints token and time stats and writes the normalized transcript.
3. Treat the current folder as the project boundary. Stop with a clear message when no matching sessions exist.
4. Read `/tmp/effort-map-sessions.jsonl` in chunks. Collect every `user` prompt verbatim (trim noise, drop tool-result echoes and system reminders).
5. Score each metric from 0 to 100 from the evidence in the transcript, and write one honest sentence of WHY for each:
   - **effort** — reasoning, planning, and iteration versus one-shotting.
   - **bugs** — time and turns spent chasing and fixing bugs.
   - **experience** — tuning UX, polish, and details versus generate-and-done.
   - **architecture** — care about stack, structure, and design decisions.
   - **copy_slop** — copying other solutions with no changes or taste (high = more slop).
6. Merge the stats from `/tmp/effort-map-stats.json` (`avg_tokens`, `total_tokens`, `avg_time_minutes`, `sessions_matched`). Do not invent token or time numbers; use the ones the collector computed.
7. Write `/tmp/effort-map-findings.json`:
   ```json
   {
     "agent": "Claude",
     "project": "project-name",
     "generated": "2026-07-06",
     "sessions_analyzed": 12,
     "prompts": ["first prompt", "second prompt"],
     "stats": { "avg_tokens": 45231, "total_tokens": 542772, "avg_time_minutes": 23.4 },
     "metrics": {
       "effort":       {"score": 72, "why": "..."},
       "bugs":         {"score": 40, "why": "..."},
       "experience":   {"score": 65, "why": "..."},
       "architecture": {"score": 80, "why": "..."},
       "copy_slop":    {"score": 15, "why": "..."}
     }
   }
   ```
8. Render the report. From this skill directory run:
   ```
   python3 scripts/render_report.py --input /tmp/effort-map-findings.json --output-dir "$PWD"
   ```
9. Return the absolute report path, the five scores, and the session count.

## Scoring Rules

- Score from evidence only. Base each number on what the transcript shows, not on a guess about the user.
- Keep every WHY to one specific sentence that names what happened.
- Remove secrets, credentials, and personal data from prompts before writing them.
- Never modify the project while scoring. The only file written into the project is the HTML report.
