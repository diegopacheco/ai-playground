# cc-hook-tool-time-tracker

A Claude Code hook that measures and logs how long each tool call takes.

## How It Works

Two hooks fire on every tool invocation:

1. **PreToolUse** — `track-start.sh` records the start timestamp to a temp file
2. **PostToolUse** — `track-end.sh` reads the start time, calculates elapsed ms, appends a JSONL entry to the log

All timing data is written to `~/.claude-hooks/tool-time-tracker.log` in JSONL format:

```json
{"timestamp":"2026-03-20T10:15:30Z","session_id":"abc123","tool":"Bash","elapsed_ms":1523,"cwd":"/path/to/project"}
```

## Requirements

- **jq** — `brew install jq`
- **perl** — pre-installed on macOS (used for millisecond timestamps)
- **bash** — standard shell

## Install

```bash
chmod +x install.sh
./install.sh
```

This copies the hook scripts to `~/.claude-hooks/cc-tool-time-tracker/` and patches `~/.claude/settings.json` to register the hooks. A backup of settings.json is created before patching.

## Uninstall

```bash
chmod +x uninstall.sh
./uninstall.sh
```

Removes the hook scripts and entries from settings.json. The log file is preserved (printed path so you can delete it manually).

## Usage

Once installed, timing is automatic. Every tool call in Claude Code gets logged.

### View Report

```bash
./scripts/report.sh
```

Prints a per-tool summary table with count, avg, min, max, and total elapsed time, plus the last 10 log entries.

```
=== CC Tool Time Tracker Report ===
Log file: /Users/you/.claude-hooks/tool-time-tracker.log
Entries: 142 | Size: 24K

--- Per-Tool Summary (sorted by total time) ---
TOOL                    COUNT    AVG(ms)    MIN(ms)    MAX(ms)     TOTAL(ms)
----                    -----    -------    -------    -------     ---------
Bash                       45       2340        120      15200        105300
Read                       38        180         50        890          6840
Edit                       22        310         80       1200          6820
Grep                       19        250         60        980          4750
Write                      10        420        150       1100          4200
Glob                        8        140         30        450          1120
```

### Clear Log

```bash
./scripts/clear-log.sh
```

## File Structure

```
cc-hook-tool-time-tracker/
├── design-doc.md          # architecture and design decisions
├── README.md
├── hooks/
│   ├── track-start.sh     # PreToolUse hook — records start time
│   └── track-end.sh       # PostToolUse hook — calculates and logs elapsed time
├── scripts/
│   ├── report.sh          # pretty-print timing summary
│   └── clear-log.sh       # wipe the log file
├── install.sh             # install hooks + patch settings.json
└── uninstall.sh           # remove hooks + patch settings.json
```

## Log Location

- **Timing log**: `~/.claude-hooks/tool-time-tracker.log`
- **Temp state files**: `/tmp/cc-tool-timing/` (auto-cleaned on reboot)
- **Settings backup**: `~/.claude/settings.json.bak`

## Sample Result

```
❯ could you run my time tracker report?

⏺ Bash(bash /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/cc-hook-tool-time-tracker/scripts/report.sh)
⎿  === CC Tool Time Tracker Report ===
    Log file: /Users/diegopacheco/.claude-hooks/tool-time-tracker.log
    Entries: 4 | Size: 4.0K

    --- Per-Tool Summary (sorted by total time) ---
    TOOL                    COUNT    AVG(ms)    MIN(ms)    MAX(ms)    TOTAL(ms)
    ----                    -----    -------    -------    -------    ---------
    Bash                        4        353         87       1146         1412

    --- Last 10 Entries ---
    2026-03-20T20:16:27Z | Bash | 1146ms
    2026-03-20T20:16:30Z | Bash | 87ms
    2026-03-20T20:16:31Z | Bash | 91ms
    2026-03-20T20:16:44Z | Bash | 88ms
⎿  2 PostToolUse hooks ran

                                                                                                        01:16 PM claude-opus-4-6
⏺ Here's your report:

- 4 tool calls tracked, all Bash
- Average: 353ms
- Fastest: 87ms
- Slowest: 1,146ms
- Total time: 1,412ms

Looks like the hooks are working and tracking tool execution times.
```