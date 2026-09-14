---
name: log-summarizer
description: Summarizes the last hour of application logs into the top errors. Use when the user asks what went wrong recently.
---

# Log Summarizer

Run `python3 scripts/summarize.py <log-file>` and report the five most frequent errors with a
count and the first time each one appeared.
