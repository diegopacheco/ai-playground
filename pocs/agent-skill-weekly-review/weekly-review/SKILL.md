---
name: weekly-review
description: Produce a flashy light-theme website that recaps the past week of work by combining the current git repository's history with the user's Google Calendar events. Use when the user runs /weekly-review or asks for a weekly summary, recap, or report of their work.
---

# Weekly Review

Build a polished, flashy, light-theme single-page website that summarizes the last week of work from two sources: the git history of the current repository and the user's Google Calendar.

The generator is pure Python standard library (no install step beyond `install.sh`) and lives at:

```
~/.claude/skills/weekly-review/scripts/weekly_review.py
```

## Steps

1. **Date range.** Default to the last 7 days (today minus 6 through today). If the user names a range, pass `--since YYYY-MM-DD --until YYYY-MM-DD`.

2. **Calendar source** — the Google Calendar *secret address in iCal format*:
   - If `WEEKLY_REVIEW_ICAL_URL` is set, use it.
   - Else if `~/.claude/skills/weekly-review/calendar.url` exists, read the URL from it.
   - Else ask the user for their secret iCal address (Google Calendar → Settings → *Settings for my calendars* → pick calendar → *Integrate calendar* → *Secret address in iCal format*) and offer to save it to `~/.claude/skills/weekly-review/calendar.url`. If they decline, continue git-only.
   - The secret URL is sensitive. Never print it back and never commit it.

3. **Generate** from the repository being reviewed:
   ```
   python3 ~/.claude/skills/weekly-review/scripts/weekly_review.py \
     --repo . --ical-url "$URL" --out ./weekly-review-site
   ```
   Use `--ical-file PATH` for a local `.ics` instead of a URL. Omit both for a git-only review. Add `--author "Name"` to focus on one contributor.

4. **Highlights.** The command prints a JSON summary on stdout. Read it, write 2–4 punchy sentences about the week (notable work, busiest day, meeting load), and re-run with `--highlights "..."` to embed them in the hero.

5. **Open** `./weekly-review-site/index.html` (`open` on macOS, `xdg-open` on Linux) and tell the user the path.

## Output

A self-contained `index.html`: animated hero with the week's highlight line, count-up stat cards (commits, lines added/removed, files, meetings, meeting hours), a commits-per-day bar chart, a file-type donut, an activity heatmap, most-changed files, contributors, meeting-hours-per-day, longest meetings, and a merged commit + meeting timeline. No build step, no runtime dependencies.
