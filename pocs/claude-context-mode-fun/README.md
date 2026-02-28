# Claude Context Mode

https://github.com/mksglu/claude-context-mode

## Install

```
/plugin marketplace add mksglu/claude-context-mode
/plugin install context-mode@claude-context-mode
```

## Token Usage

```
⏺ Here's the full breakdown of what claude-context-mode costs you:

Session Context Usage (right now)

┌─────────────────────────┬─────────────────────────────────────┐
│         Metric          │                Value                │
├─────────────────────────┼─────────────────────────────────────┤
│ Tool calls this session │                                   2 │
├─────────────────────────┼─────────────────────────────────────┤
│ Data processed          │                             58.7 KB │
├─────────────────────────┼─────────────────────────────────────┤
│ Tokens consumed         │                             ~15,016 │
├─────────────────────────┼─────────────────────────────────────┤
│ Context savings         │ 0% (no large outputs processed yet) │
└─────────────────────────┴─────────────────────────────────────┘

Disk Footprint

┌────────────────────────────────────┬─────────┬────────┬────────────┐
│             Component              │  Size   │ Lines  │ Characters │
├────────────────────────────────────┼─────────┼────────┼────────────┤
│ Full plugin directory              │ 1.57 MB │ 21,838 │ 1,585,720  │
├────────────────────────────────────┼─────────┼────────┼────────────┤
│ Hooks (pretooluse.sh + hooks.json) │ 6.6 KB  │ 199    │ 6,654      │
├────────────────────────────────────┼─────────┼────────┼────────────┤
│ SKILL.md files (4 skills)          │ 18.1 KB │ 363    │ 18,196     │
├────────────────────────────────────┼─────────┼────────┼────────────┤
│ .mcp.json                          │ 126 B   │ 8      │ 126        │
└────────────────────────────────────┴─────────┴────────┴────────────┘

What It Injects Into Every Conversation

The constant overhead per session comes from:

1. 7 MCP tool definitions — execute, execute_file, index, search, fetch_and_index, batch_execute, stats — each with full JSON
schema descriptions. These are loaded into your system prompt every turn.
2. 4 skill descriptions in the system reminder (context-mode, stats, doctor, upgrade) — ~4 lines each.                             3. PreToolUse hooks — run on every Bash, Read, Grep, WebFetch, and Task call to potentially intercept/redirect them.
                                                                                                                                    The Tradeoff

- Cost: ~2-3K tokens of tool definitions added to every API call, plus hook execution overhead on every tool call.
- Benefit: When you process large outputs (logs, test results, git history, Playwright snapshots), it keeps raw data out of
context — saving 95-99% on those calls. The plugin pays for itself once you start hitting large outputs.

Right now with 0% savings, it's pure overhead. It shines when you're doing heavy research, running tests, or analyzing large
files.
```