---
name: market-research
description: Research a market or product with the market_report tool. Use when the user asks for a market report, market research, competitor scan or trend analysis.
---

# Market research

1. Pick the topic in 2 to 6 words, such as "electric bikes europe".
2. Use depth "quick" for a first look and "deep" for a full report.
3. Call the `market_report` tool. It is long running: it returns a task, keep working while it runs.
4. When the task completes, format the result with the rules in [report-style](references/report-style.md).
5. If the user changes their mind, cancel the task instead of waiting for it.
