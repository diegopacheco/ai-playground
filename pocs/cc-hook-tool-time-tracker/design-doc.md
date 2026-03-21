# cc-hook-tool-time-tracker — Design Doc

## Overview

A Claude Code hook that measures and logs how long each tool call takes. It uses the `PreToolUse` and `PostToolUse` hook events to capture start/end timestamps, calculates elapsed time, and writes structured metrics to a log file.

## Problem

There is no built-in way to see how long each tool invocation takes in Claude Code. When debugging slow sessions or optimizing workflows, knowing which tools are slow and how time is distributed across tool calls is valuable.

## Solution

Two bash scripts registered as Claude Code hooks:

- **`track-start.sh`** — Fires on `PreToolUse`, records tool name + start timestamp to a temp state file
- **`track-end.sh`** — Fires on `PostToolUse`, reads the start timestamp, calculates elapsed time, appends a structured log entry

### Architecture

```
┌──────────────┐     stdin (JSON)     ┌─────────────────┐
│  Claude Code  │ ──────────────────► │  track-start.sh │
│  PreToolUse   │                     │  writes state    │
└──────────────┘                      └────────┬────────┘
                                               │
                                     /tmp/cc-tool-timing/
                                     {session_id}_{tool}.start
                                               │
┌──────────────┐     stdin (JSON)     ┌────────▼────────┐
│  Claude Code  │ ──────────────────► │  track-end.sh   │
│  PostToolUse  │                     │  reads state     │
└──────────────┘                      │  calculates dt   │
                                      │  appends log     │
                                      └────────┬────────┘
                                               │
                                     ~/.claude-hooks/
                                     tool-time-tracker.log
```

### Hook Input (stdin from Claude Code)

PreToolUse provides:
```json
{
  "session_id": "abc123",
  "tool_name": "Bash",
  "tool_input": { "command": "npm test" },
  "hook_event_name": "PreToolUse",
  "cwd": "/path/to/project"
}
```

PostToolUse provides the same fields plus `tool_output` (which we ignore for timing purposes).

### State Management

Each tool invocation writes a start-time file to `/tmp/cc-tool-timing/`:
- Filename: `{session_id}_{tool_name}.start`
- Content: epoch milliseconds

This supports concurrent tool calls by keying on session + tool name.

**Edge case:** If two calls to the same tool overlap within one session, the second `PreToolUse` overwrites the first start time. This is acceptable — it's a best-effort tracker, not a billing system. A future improvement could add a unique invocation ID if Claude Code provides one.

### Log Output

Appends JSONL entries to `~/.claude-hooks/tool-time-tracker.log`:

```json
{"timestamp":"2026-03-20T10:15:30Z","session_id":"abc123","tool":"Bash","elapsed_ms":1523,"cwd":"/path/to/project"}
```

### Hook Configuration (settings.json)

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude-hooks/cc-tool-time-tracker/track-start.sh",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude-hooks/cc-tool-time-tracker/track-end.sh",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

- Empty `matcher` matches all tools
- 5-second timeout prevents hanging if something goes wrong
- Scripts always exit 0 so they never block tool execution

## File Structure

```
cc-hook-tool-time-tracker/
├── design-doc.md
├── README.md
├── hooks/
│   ├── track-start.sh
│   └── track-end.sh
├── scripts/
│   ├── report.sh          # pretty-print timing summary from the log
│   └── clear-log.sh       # wipe the log file
├── install.sh             # copies hooks + patches settings.json
└── uninstall.sh           # removes hooks + patches settings.json
```

## Scripts Detail

### track-start.sh
1. Read JSON from stdin using `jq`
2. Extract `session_id` and `tool_name`
3. Create `/tmp/cc-tool-timing/` if missing
4. Write current epoch ms to `/tmp/cc-tool-timing/{session_id}_{tool_name}.start`
5. Exit 0

### track-end.sh
1. Read JSON from stdin using `jq`
2. Extract `session_id`, `tool_name`, `cwd`
3. Read start time from `/tmp/cc-tool-timing/{session_id}_{tool_name}.start`
4. Calculate elapsed ms
5. Append JSONL entry to `~/.claude-hooks/tool-time-tracker.log`
6. Remove the `.start` file
7. Exit 0

### report.sh
1. Read `~/.claude-hooks/tool-time-tracker.log`
2. Aggregate by tool name: count, avg, min, max, total
3. Print a formatted table sorted by total time descending

### install.sh
1. Create `~/.claude-hooks/cc-tool-time-tracker/`
2. Copy `track-start.sh` and `track-end.sh` into it
3. Make them executable
4. Read `~/.claude/settings.json`
5. Use `jq` to merge the hook configuration (preserving existing hooks)
6. Write updated `settings.json`
7. Print success message

### uninstall.sh
1. Read `~/.claude/settings.json`
2. Use `jq` to remove the time-tracker hook entries
3. Write updated `settings.json`
4. Remove `~/.claude-hooks/cc-tool-time-tracker/`
5. Optionally remove the log file (ask or flag)
6. Print success message

## Dependencies

- **jq** — required for JSON parsing in bash. This is the only external dependency.
- **bash** — standard shell
- **date** — for epoch millisecond timestamps (`gdate` on macOS if needed, or `python3 -c` fallback)

### macOS Timestamp Note

macOS `date` does not support `%N` (nanoseconds). Options:
1. Use `python3 -c "import time; print(int(time.time()*1000))"` — available on all macOS
2. Use `gdate` from coreutils if installed
3. Use millisecond precision from `perl -MTime::HiRes`

The scripts will use the `perl` approach since it's pre-installed on macOS and fast:
```bash
perl -MTime::HiRes=time -e 'printf "%d\n", time*1000'
```

## Trade-offs and Decisions

| Decision | Why |
|----------|-----|
| JSONL log format | Easy to append, easy to parse line-by-line, grep-friendly |
| `/tmp` for state files | Ephemeral, auto-cleaned on reboot, fast |
| `perl` for timestamps | Pre-installed on macOS, no extra dependencies, sub-ms precision |
| Per-tool state files | Supports concurrent tool calls without a database |
| Always exit 0 | Time tracking must never block or interfere with Claude Code |
| Install to `~/.claude-hooks/` | User-scoped, does not pollute project directories |
| Merge into existing settings | Preserves user's other hooks and configuration |

## Risks

1. **jq not installed** — install.sh should check and warn. The hook scripts should also fail gracefully (exit 0) if jq is missing.
2. **settings.json merge conflicts** — if user has a complex hook setup, the jq merge could collide. install.sh should back up settings.json before modifying.
3. **Disk space** — log file grows unbounded. report.sh should show file size. clear-log.sh provides manual cleanup. A future improvement could add log rotation.
4. **Concurrent same-tool calls** — start file gets overwritten. Acceptable for v1 since this is rare and the error is just slightly inaccurate timing for one entry.

## Future Improvements (Out of Scope for v1)

- Log rotation (e.g., keep last 10MB or last 7 days)
- Real-time terminal dashboard (`watch` + report.sh)
- Per-project log separation
- Export to CSV/JSON summary
- Hook for `PostToolUseFailure` to track failed tool timing separately
