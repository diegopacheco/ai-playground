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

## More info...

```
❯ how do I know if claude-context-mode is really saving tokens for me? can you figure out?

⏺ plugin:context-mode:context-mode - stats (MCP)
  ⎿  ## context-mode session stats

     | Metric | Value |
     … +11 lines (ctrl+o to expand)

⏺ Context-mode wasn't used this session - all the work was done with the standard Read, Edit, Write, Bash, and Agent tools. The
  savings would show up if we had used batch_execute, execute_file, or fetch_and_index instead of Read/Bash for large outputs (like
   the GraalVM build output, git logs, etc.).

  In this session the largest outputs came from your terminal pastes (build logs), not from tool calls, so context-mode wouldn't
  have helped much here. It's most useful when you're running commands that produce large output (test suites, log analysis,
  dependency trees) - those get processed in a sandbox and only a summary enters context.

❯ how many tokens context-mode loads into my context window, chars and lines not matter if I use or not?

⏺ Based on the /context output you ran, context-mode costs:

  MCP tools: 2,200 tokens (7 tools at ~310 tokens each)

  Plus the skills:
  - context-mode skill: 306 tokens
  - stats: 46 tokens
  - upgrade: 44 tokens
  - doctor: 40 tokens

  Total: ~2,636 tokens loaded just by having context-mode registered, regardless of whether you use it or not.

  That's ~1.3% of your 200k context window. Not huge, but it's there on every conversation. If you're not actively using it, those
  tokens are wasted overhead.
```

## Savings

```
⏺ context-mode saved you 62% of context space this session.

  ┌─────────────────────────────────────────┬──────────────────────┐
  │                 Metric                  │        Value         │
  ├─────────────────────────────────────────┼──────────────────────┤
  │ Total data processed                    │             453.1 KB │
  ├─────────────────────────────────────────┼──────────────────────┤
  │ Kept in sandbox (never entered context) │             282.4 KB │
  ├─────────────────────────────────────────┼──────────────────────┤
  │ Actually entered context                │             170.8 KB │
  ├─────────────────────────────────────────┼──────────────────────┤
  │ Tokens consumed                         │              ~43,717 │
  ├─────────────────────────────────────────┼──────────────────────┤
  │ Tokens saved                            │              ~72,284 │
  ├─────────────────────────────────────────┼──────────────────────┤
  │ Savings ratio                           │ 2.7x (62% reduction) │
  └─────────────────────────────────────────┴──────────────────────┘

  Without context-mode, all 453 KB of git logs, diffs, file contents, and GitHub fetches would have flooded your context window.
  Instead, 282 KB stayed in the sandbox and only the relevant search results came through. That's ~72K tokens you didn't burn
  reading raw git output.
```