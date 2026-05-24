# cc-token-bar — Design Document

A macOS menu bar (status bar) app that visualizes Claude Code token usage and cost, fed by hooks that Claude Code calls into.

---

## 1. Goal

Give a Claude Code user a permanently visible, glanceable readout of:

- Total tokens consumed (lifetime + today).
- Token usage over time (week, all time) as a chart.
- Cost per tool, listed and sorted.

Data is captured locally via Claude Code hooks and stored as JSON under `~/.cc-token-bar/`. Nothing leaves the machine.

## 2. Non-goals

- No remote sync, no cloud, no telemetry.
- No editing or replaying of transcripts.
- No real-time per-token streaming readout (per-session granularity is enough).
- No support for non-macOS platforms.
- No support for other AI CLIs (Codex, Gemini CLI, etc.) in v1.

## 3. User stories

- As a user I open the menu bar icon and see today's tokens and lifetime tokens at the top.
- I see a 7-day bar chart of input vs output tokens.
- I see a list of tools (Read, Edit, Bash, …) ranked by total cost, with invocation counts.
- I install with one script and uninstall cleanly with another.

## 3.5. What's already in `~/.claude/` (don't reinvent)

Before designing storage, here's what Claude Code already writes. Most of our work is *reading*, not *capturing*.

| Source                                              | What it gives us                                                                                                                          | Use it for                                          |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `~/.claude/.session-stats.json`                     | Per-session tool counts, `started_at`, `updated_at`, `total_calls`, `last_tool`                                                           | Tool list, session length, sessions/day             |
| `~/.claude/stats-cache.json`                        | Pre-aggregated daily activity: `messageCount`, `sessionCount`, `toolCallCount` per date                                                   | Long-term daily chart (no recomputation needed)     |
| `~/.claude/projects/<encoded-path>/<session>.jsonl` | Full transcript. Per assistant message: `usage.input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `model`, `service_tier`, `server_tool_use.web_search_requests`, `web_fetch_requests`, `cache_creation.ephemeral_1h_input_tokens` vs `ephemeral_5m_input_tokens` | Token totals, cost, cache efficiency, web tool cost |
| `~/.claude/history.jsonl`                           | Every prompt with `timestamp` + `project` (4 500+ entries on this machine)                                                                | Prompts/day, prompts/project, activity heatmap      |
| `~/.claude/projects/<encoded-path>/`                | Directory name encodes project path → group by project                                                                                    | Per-project breakdown                               |
| `~/.claude/file-history/<session>/<hash>@vN`        | File snapshots per session                                                                                                                | Files touched per session                           |
| `~/.claude/sessions/<pid>.json`                     | Live session metadata                                                                                                                     | "Active now" indicator                              |

**Implication:** the hook only needs to *poke* the app to refresh (and optionally maintain a small incremental index). All raw data already exists. This is a big simplification — see §5.

## 4. Architecture

```
            ~/.claude/  (Claude Code writes these — we read)
            ├── .session-stats.json     ← tool counts per session
            ├── stats-cache.json        ← daily messages/sessions/tools
            ├── history.jsonl           ← every prompt
            └── projects/<enc>/<sid>.jsonl  ← transcripts w/ usage
                                  │
                                  │  FSEvents on this tree
                                  ▼
+-------------------+   poke    +--------------------------------+
|   Claude Code     |  ──────▶  |  cc-token-bar-hook.sh (Stop)  |
+-------------------+           |  updates incremental index    |
                                +---------------+----------------+
                                                │
                                                ▼
                                +--------------------------------+
                                |   ~/.cc-token-bar/             |
                                |     index.json   (rollups)     |
                                |     state/offsets.json         |
                                |     config.json  (pricing)     |
                                +---------------+----------------+
                                                │
                                                ▼
                                +--------------------------------+
                                |   cc-token-bar.app             |
                                |   SwiftUI MenuBarExtra         |
                                |   reads ~/.claude + own index  |
                                +--------------------------------+
```

Three pieces:

1. **Hook shim** *(optional but recommended)* — invoked on `Stop` / `SessionEnd`. Reads only the *new* bytes of the active transcript (via saved offset) and updates `index.json`. Without it the app still works by scanning on its own; the hook just keeps things instant.
2. **Storage** — only *derived* rollups live in `~/.cc-token-bar/`. Raw truth stays in `~/.claude/`.
3. **Menu bar app** — SwiftUI `MenuBarExtra`, watches both trees with FSEvents, renders totals + chart + tool list.

## 5. Data capture (hooks)

Claude Code hook events used:

| Event          | Why                                                                 |
| -------------- | ------------------------------------------------------------------- |
| `PostToolUse`  | Increment a per-tool invocation counter (cheap, no transcript scan) |
| `Stop`         | Session turn complete: rescan transcript, recompute session totals  |
| `SessionEnd`   | Final flush                                                         |

Both events receive a JSON payload on stdin that includes `session_id` and `transcript_path`.

The hook shim:

1. Reads stdin JSON → extracts `session_id`, `transcript_path`, `tool_name` (for `PostToolUse`).
2. Streams the transcript JSONL and sums `message.usage.*` per model.
3. Writes/overwrites `~/.cc-token-bar/sessions/<session_id>.json`.
4. For `PostToolUse`: appends/updates `~/.cc-token-bar/tools/<session_id>.json` with the per-tool counter and approximate input/output bytes from `tool_input` / `tool_response`.

Hook registration lives in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "~/.cc-token-bar/bin/hook.sh post_tool_use" }] }],
    "Stop":        [{ "matcher": "",  "hooks": [{ "type": "command", "command": "~/.cc-token-bar/bin/hook.sh stop" }] }],
    "SessionEnd":  [{ "matcher": "",  "hooks": [{ "type": "command", "command": "~/.cc-token-bar/bin/hook.sh session_end" }] }]
  }
}
```

The hook must finish in well under the default hook timeout (60 s). Worst case is a long transcript — mitigation: incremental scan using a saved byte offset per session.

## 6. Storage layout

```
~/.cc-token-bar/
├── bin/
│   └── hook.sh
├── config.json                       # pricing table, refresh interval
├── sessions/
│   └── <session_id>.json             # one per Claude Code session
├── tools/
│   └── <session_id>.json             # one per session, per-tool counters
└── state/
    └── offsets.json                  # last-read byte offset per transcript
```

**Session timestamps:** `started_at` is the minimum `timestamp` across all records in the transcript JSONL; `updated_at` is the maximum. Falls back to the transcript file's mtime if no record carries a timestamp. This matters for backfill — using `now` would attribute every historical session to today and the daily chart would collapse to a single bar.

**Number formatting:** tokens use `B` / `M` / `k` suffixes (`537.1M`, `12k`). USD uses `NumberFormatter(.currency)` so it renders with the thousands separator (`$1,344.62`). Costs above 1M USD switch to `$N.NNM` to keep the column narrow.

### `sessions/<session_id>.json`

```json
{
  "session_id": "01HXY...",
  "project_path": "/Users/diego/git/foo",
  "started_at": "2026-05-23T17:25:00Z",
  "updated_at": "2026-05-23T18:10:42Z",
  "by_model": {
    "claude-opus-4-7": {
      "input_tokens": 12034,
      "output_tokens": 4310,
      "cache_creation_input_tokens": 22100,
      "cache_read_input_tokens": 188430
    }
  }
}
```

### `tools/<session_id>.json`

```json
{
  "session_id": "01HXY...",
  "tools": {
    "Read":  { "count": 42, "input_bytes": 8120,  "output_bytes": 412300 },
    "Edit":  { "count": 11, "input_bytes": 5430,  "output_bytes":  12000 },
    "Bash":  { "count":  7, "input_bytes":  610,  "output_bytes":   8400 }
  }
}
```

### `config.json`

Pricing table, refresh interval, and which models to show. User-editable.

## 7. Cost model

Per-model pricing (USD per 1M tokens), seeded at install, editable in `config.json`:

| Model           | Input | Output | Cache write | Cache read |
| --------------- | ----: | -----: | ----------: | ---------: |
| Opus 4.x        | 15.00 |  75.00 |       18.75 |       1.50 |
| Sonnet 4.x      |  3.00 |  15.00 |        3.75 |       0.30 |
| Haiku 4.5       |  1.00 |   5.00 |        1.25 |       0.10 |

`cost = (input * pIn + output * pOut + cache_create * pCW + cache_read * pCR) / 1_000_000`

**"Cost per tool"** is computed by attribution — see open question Q3.

## 7.5. Metrics catalog (what we can show)

Beyond "total tokens" and "tools by cost", here's the broader set that's cheap to derive from existing data. You pick which land in v1.

**Cost & tokens**
- Lifetime / 7-day / today: input, output, cache-write, cache-read tokens.
- USD cost using `config.json` pricing.
- **Cache hit ratio** = `cache_read / (cache_read + cache_creation + input)`. Single number that tells you how efficient your sessions are. Likely the most actionable metric here.
- **Cache tier split**: ephemeral 1h vs 5m token shares.
- **Cost by model** (Opus vs Sonnet vs Haiku) — direct lever for spend.
- **Server-tool cost**: `web_search_requests`, `web_fetch_requests` are billed separately from tokens. Worth its own row.
- **Thinking-token share** — how much you spend on extended thinking vs final output.

**Activity (from `stats-cache.json` + `history.jsonl`)**
- Prompts per day (history.jsonl), sessions per day, tool calls per day.
- Average prompts per session.
- Activity heatmap (GitHub-style, 12-week grid).
- "Day streak" (consecutive days with at least one session).

**Tools (from `.session-stats.json` + transcripts)**
- Top tools by count.
- Top tools by **estimated cost** (see Q3 in open questions for definitions).
- Most-used skills (from `skill_listing` records in transcripts).
- Subagent (Task) spawn count.
- Ultrathink usage count (from `ultrathink_effort` records).

**Per-project (from project directory names)**
- Tokens & cost per project, sortable.
- Sessions per project.
- Most expensive project this week.

**Live**
- Active session indicator (from `~/.claude/sessions/`).
- "Tokens this session so far" (tail of the active transcript).

## 8. UI — menu bar app

**Stack:** Swift + SwiftUI, `MenuBarExtra` (macOS 13+). No third-party libraries. Chart drawn with Swift Charts (system framework).

**Menu bar item:** fixed-width identifier — a bar-chart SF Symbol followed by the literal text `cc`. The label is intentionally static to minimise menu bar real estate and survive crowded notch layouts on MacBooks. All dynamic numbers live inside the dropdown.

**Implementation:** `NSStatusItem` + `NSPopover` hosting a `SwiftUI` `PanelView` via `NSHostingController` (MenuBarExtra proved unreliable on macOS 26 when built via SwiftPM — see commit history). The status item is created in `AppDelegate.applicationDidFinishLaunching` after `NSApp.setActivationPolicy(.accessory)`.

**Live updates:** the panel auto-refreshes within ~1 second of any change under `~/.cc-token-bar/`. Flow: hook writes a new aggregate via atomic rename → `FSEventStreamCreate` with `kFSEventStreamCreateFlagFileEvents` fires → `DataStore.scheduleRefresh()` debounces 250 ms → JSON reload off the main queue → `@Published agg` updates on main → SwiftUI re-renders the open popover. A 30 s polling timer is a fallback in case any event is missed.

Click opens this panel:

```
┌──────────────────────────────────────────┐
│  cc-token-bar                       ⚙ ✕  │
├──────────────────────────────────────────┤
│  Today        412,103 tokens   $1.84     │
│  All-time   8,910,442 tokens  $42.17     │
├──────────────────────────────────────────┤
│  Last 7 days                             │
│   ▆ ▃ ▅ ▇ ▂ ▄ █     in / out stacked    │
│   M  T  W  T  F  S  S                    │
├──────────────────────────────────────────┤
│  Tools (by cost)                         │
│   Read   ████████░  $14.20   (1,204×)    │
│   Edit   █████░░░░   $8.40   (  311×)    │
│   Bash   ██░░░░░░░   $3.10   (  192×)    │
│   ...                                    │
├──────────────────────────────────────────┤
│  Open data folder    Quit                │
└──────────────────────────────────────────┘
```

**Refresh:** FSEvents watcher on `~/.cc-token-bar/sessions/` and `tools/`. Recompute aggregates on change. Fallback: 30 s timer.

**Aggregation cost:** scanning hundreds of small JSON files is cheap; if it ever isn't, add a `rollup/daily/<YYYY-MM-DD>.json` cache.

## 9. Install / uninstall

### `install.sh`

1. Create `~/.cc-token-bar/{bin,sessions,tools,state}`.
2. Copy `hook.sh` into `~/.cc-token-bar/bin/`, `chmod +x`.
3. Write default `config.json` if missing.
4. Back up `~/.claude/settings.json` → `~/.claude/settings.json.bak.<timestamp>`.
5. Merge hook entries into `~/.claude/settings.json` with `jq` (idempotent: skip if entries already present).
6. Build the Swift app (`xcodebuild` or `swift build`) and copy `cc-token-bar.app` to `/Applications/`.
7. Register a LaunchAgent (`~/Library/LaunchAgents/com.cc-token-bar.plist`) for autostart — opt-in via flag.
8. Launch the app.

### `uninstall.sh`

1. Stop running app, unload LaunchAgent, delete plist.
2. Remove hook entries from `~/.claude/settings.json` with `jq` (leave a backup).
3. Remove `/Applications/cc-token-bar.app`.
4. Prompt before deleting `~/.cc-token-bar/` (user may want to keep the data).

Both scripts: no comments, no `sleep > 1`, no emojis (per project conventions).

## 10. HTML preview

`preview.html` (sibling file) mocks the menu bar dropdown with fake data: header totals, inline SVG 7-day chart, tool-cost list. No JS libraries — pure HTML/CSS/SVG. Use this to iterate on layout before any Swift is written.

## 11. Decisions (locked) & remaining questions

### Locked

| # | Topic                  | Decision                                                                                                  |
| - | ---------------------- | --------------------------------------------------------------------------------------------------------- |
| 1 | macOS target           | macOS 13+ (`MenuBarExtra`)                                                                                |
| 2 | Stack                  | Native Swift + SwiftUI, no third-party libraries                                                          |
| 3 | Cost-per-tool semantics | Tokens **and** $ per tool. Attribution = tool-result bytes (option (a), cleanest defensible number)        |
| 4 | Hook                   | Ship the hook, mark optional in `install.sh`. App falls back to FSEvents-only if hook not installed       |
| 5 | v1 metric set          | Today + lifetime header · 7-day stacked chart · tools by cost · cache hit ratio · cost by model split     |
| 6 | Backfill on install    | Yes — scan all existing `~/.claude/projects/**/*.jsonl` and seed `index.json`                             |

### All locked (defaults accepted)

| #  | Topic                              | Decision                                                                                                  |
| -- | ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 5  | Per-project breakdown in v1        | Defer to v2; index keeps `project_path` so it lights up later for free                                    |
| 6  | Auto-launch on login               | Opt-in via `install.sh --autostart` flag; creates a LaunchAgent plist                                     |
| 7  | Pricing maintenance                | Hard-code current prices in `config.json`, user-editable. No fetch                                        |
| 8  | Concurrent CC sessions on offsets  | Write `offsets.json.tmp` then atomic `mv`                                                                 |
| 9  | Subagent (Task) transcripts        | Treated as their own sessions, tagged `parent_session_id` so v2 can roll them up                          |
| 10 | Privacy                            | Store `project_path` in the index (no message bodies). `install.sh --no-paths` available for opt-out      |
| 13 | Install/uninstall responsibilities | `install.sh` handles everything end-to-end (dir layout, hook, settings.json merge, app bundle, LaunchAgent, backfill). `uninstall.sh` mirrors it exactly in reverse |

No remaining open questions.
