---
name: logs
description: Read log files and summarize errors, their first occurrence and their likely cause. Use when the user asks what is wrong in the logs, or runs /troubleshoot:logs.
---

# Logs

Read the logs given in the arguments: `$ARGUMENTS`. When no path is given, look for `*.log` files and `logs/` folders in the current project.

1. Group lines by level and keep only WARN, ERROR and FATAL.
2. Collapse repeated messages into one entry with a count.
3. For each distinct error, find its first occurrence and the lines just before it.
4. Name the likely cause in one sentence.

Report a table with the error, the count, the first timestamp and the likely cause, ordered by first occurrence.
