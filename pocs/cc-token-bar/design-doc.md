# cc-token-bar — Design Document

A macOS menu bar (status bar) app that visualizes Claude Code token usage and cost, fed by hooks that Claude Code calls into.

---

## 1. Goal

Give a Claude Code user a permanently visible, glanceable readout of:

- Total tokens consumed (lifetime + today).
- Token usage over time (week, all time) as a chart.
- Cost per tool, listed and sorted.
- Latency per tool call, listed and sorted.

The dropdown is split into two tabs: **Cost** (default) and **Latency**.

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

1. **Hook shim** *(optional but recommended)* — invoked on `Stop` / `SessionEnd` / `PostToolUse`. Writes `~/.cc-token-bar/sessions/<sid>.json` (session totals snapshot) and `~/.cc-token-bar/tools/<sid>.json` (per-tool counters). The session snapshot is only a cache: the app re-reads transcripts itself so live sessions still update between Stop events.
2. **Storage** — only *derived* rollups live in `~/.cc-token-bar/`. Raw truth stays in `~/.claude/projects/<enc>/<sid>.jsonl` and is re-read on each refresh by `TranscriptScanner`.
3. **Menu bar app** — SwiftUI panel, watches `~/.cc-token-bar/` with FSEvents, scans `~/.claude/projects/` for live transcript usage on every refresh, renders totals + chart + tool list.

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

**ISO-8601 parsing gotcha:** Claude Code writes transcript timestamps with fractional seconds (`2026-05-23T21:40:06.341Z`). `ISO8601DateFormatter()` with default options *silently returns nil* for that format. The app must configure `.formatOptions = [.withInternetDateTime, .withFractionalSeconds]` (or try both formatters in sequence). Without this, every session's `updated_at` parses as nil → falls back to `todayKey` → Today equals All-time and the 7-day chart goes blank. This was a real bug — see `DataStore.parseISO(_:)`.

**Number formatting:** tokens use `B` / `M` / `k` suffixes (`537.1M`, `12k`). USD uses `NumberFormatter(.currency)` so it renders with the thousands separator (`$1,344.62`). Costs above 1M USD switch to `$N.NNM` to keep the column narrow. Session counts and tool invocation counts use `NumberFormatter(.decimal)` with grouping so `1204` renders as `1,204`.

**Synthetic-model filter:** Claude Code attaches a placeholder model name like `<synthetic>` (and other angle-bracket-wrapped markers) to internal non-API events such as hook responses and context bookkeeping. These carry no billable usage and would pollute the "Cost by model" list, so any model name matching `hasPrefix("<")` or containing `"synthetic"` is skipped during aggregation.

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
- Top tools by **average latency** — `tool_result.timestamp − tool_use.timestamp` per call, averaged per tool. Shown in the Latency tab.
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

**Tabs.** Directly under the header a segmented control switches the panel body between two tabs (`PanelView.tab`, a `@State PanelTab`):

- **Cost** *(default, shown on every open)* — the full metric set: Today/All-time KPIs, cache hit ratio, 7-day chart, tools by cost, cost by model.
- **Latency** — one row per tool, the average wall-clock latency of its calls, sorted slowest first, with a proportional bar and the call count.

The header, tab control, and footer are shared; only the body between them swaps. The selected tab is local UI state and resets to **Cost** when the app relaunches.

**Tool-name aggregation.** Playwright's MCP server exposes ~25 separate tools (`mcp__playwright__browser_click`, `mcp__playwright__browser_navigate`, …). Listed individually they bury the rest of the tool list and none is individually meaningful. `ToolMetrics.normalizeToolName(_:)` collapses every `mcp__playwright__*` name to a single `mcp_playwright` row. The collapse runs in both tabs — over the hook-written per-tool counters for the Cost tab, and over the transcript-derived latency samples for the Latency tab — so the two tabs always show the same tool identities.

**Latency derivation — from transcripts, not hooks.** Per-call latency is read straight from `~/.claude/projects/<enc>/<sid>.jsonl`, consistent with "read transcripts directly" (see §8 source-of-truth note). Each assistant `tool_use` block carries an `id` and the record's `timestamp`; the following user `tool_result` block carries the matching `tool_use_id` and its own record `timestamp`. Latency for that call = `result.timestamp − use.timestamp`. `TranscriptScanner` emits an ordered list of `ToolEvent`s while it streams the transcript and `ToolMetrics.pairLatencies(_:)` matches uses to results by id, normalises the tool name, and accumulates `count` + `totalMs` per tool into `SessionFile.tool_latency`. Orphan results (no matching use in this transcript) and negative deltas are discarded. `DataStore` sums these across sessions and divides for the displayed average. No hook changes are needed — the existing `PostToolUse` counters carry no timing.

**Implementation:** `NSStatusItem` + `NSPopover` hosting a `SwiftUI` `PanelView` via `NSHostingController` (MenuBarExtra proved unreliable on macOS 26 when built via SwiftPM — see commit history). The status item is created in `AppDelegate.applicationDidFinishLaunching` after `NSApp.setActivationPolicy(.accessory)`.

**Live updates:** the panel auto-refreshes within ~1 second of any change under `~/.cc-token-bar/`. Flow: hook writes a new aggregate via atomic rename → `FSEventStreamCreate` with `kFSEventStreamCreateFlagFileEvents` fires → `DataStore.scheduleRefresh()` debounces 250 ms → reload off the main queue → `@Published agg` updates on main → SwiftUI re-renders the open popover.

**Source of truth for token totals — read transcripts directly.** The hook only rewrites `~/.cc-token-bar/sessions/<sid>.json` on `Stop` / `SessionEnd`, so during a long-running session the cached session file is stale while `~/.cc-token-bar/tools/<sid>.json` keeps incrementing on every `PostToolUse`. That caused a visible bug: with the popover open, the Tools list updated every tick but Today / All-time / the 7-day chart sat frozen until the session ended. Fix: `TranscriptScanner` reads `~/.claude/projects/<encoded-path>/<sid>.jsonl` directly on every refresh and aggregates per-model usage and `started_at`/`updated_at` itself. Results are cached per file by mtime so unchanged transcripts cost one `stat` per tick. `DataStore.mergedSessions()` merges scanned sessions over hook-written sessions, keyed by `session_id`; transcripts win when present. Net effect: every visible-refresh tick recomputes Today, All-time, by-day bars, cache hit ratio, and by-model split from live transcript bytes — same cadence as the tools section.

**Visibility-gated polling:** in addition to FSEvents, the app drives polling refreshes based on popover visibility:

- **Popover open:** `AppDelegate` is the `NSPopoverDelegate`. On `popoverDidShow` it calls `store.startVisibleRefresh()`, which installs a 5 s repeating `Timer` on the main run loop in `.common` mode (so it keeps firing while the popover tracks the mouse). Every tick calls `scheduleRefresh()`, which runs `TranscriptScanner.scan()` + `loadTools()` + aggregate off the background queue. The 5 s cadence applies to **every** section of the panel — Today, All-time, by-day chart, cache ratio, tools, by-model — because the underlying refresh now re-reads transcripts and not just the hook-written cache.
- **Popover closed:** `popoverDidClose` invalidates the timer via `store.stopVisibleRefresh()`. No background polling runs while the menu is hidden; FSEvents alone keeps the in-memory aggregates warm for the next open.
- **On click-to-open:** `togglePopover` calls `store.refreshNow()` *before* showing the popover, so the user sees fresh numbers on frame one rather than the stale snapshot from the last close. This is what covers the "not visible → refresh when the user clicks" case.

The previous 30 s fallback timer is removed: while the popover is closed there is no one to read the data, and FSEvents already covers the case where the popover is open and a file changes between 5 s ticks.

**Panel sizing.** The popover content stack has a 12-pt `.padding(.top)` and 8-pt `.padding(.bottom)` so the header and footer don't touch the rounded corners. Width is fixed at 360 pt. Height is a fixed **preferred height of 720 pt**, capped each time the panel is shown by the screen's `visibleFrame.height − menuBarGap − screenBottomMargin` (gap = 8 pt, bottom margin = 12 pt) so the panel never grows past the visible screen.

The content is wrapped in a `ScrollView(.vertical, showsIndicators: false)`. This matters because the v1 metric set fully populated (KPIs, cache ratio, 7-day chart, up to ~10 tool rows, multi-model cost split, footer) can easily exceed 640 pt and even 720 pt on shorter screens. Earlier revisions tried measuring `NSHostingController.sizeThatFits(in:)` to grow the panel to the content, but on macOS 26 with the current SwiftUI runtime that call under-reports intrinsic height for stacks containing `Chart` + many sized rows, leaving the rendered content taller than the window and clipping both the header and the footer. ScrollView side-steps the measurement entirely: the window is always tall enough to show the top of the panel, and any overflow is reachable by scroll wheel / trackpad.

`AppDelegate.showPanel()` flow:

1. Find the status item's button frame in screen coordinates.
2. `topLimit = min(buttonFrame.minY − menuBarGap, visibleFrame.maxY − menuBarGap)` — the lower bound of the menu-bar area, clamped to the screen.
3. `maxHeight = visibleFrame.height − menuBarGap − screenBottomMargin`; final height = `min(preferredPanelHeight, maxHeight)` with a 240 pt floor.
4. Center horizontally under the status button, then clamp into `[visibleFrame.minX + 8, visibleFrame.maxX − width − 8]` so the panel can't slip off-screen for a status item that lives near the right notch.
5. `setFrame(_:display:)` and `orderFrontRegardless()`.

Click opens this panel:

```
┌──────────────────────────────────────────┐
│  cc-token-bar                          ✕  │
├──────────────────────────────────────────┤
│  [   Cost   ] [  Latency  ]              │  ← segmented tab control
├──────────────────────────────────────────┤
│  Today        412,103 tokens   $1.84     │   Cost tab (default)
│  All-time   8,910,442 tokens  $42.17     │
├──────────────────────────────────────────┤
│  Last 7 days                             │
│   ▆ ▃ ▅ ▇ ▂ ▄ █     in / out stacked    │
│   M  T  W  T  F  S  S                    │
├──────────────────────────────────────────┤
│  Tools (by cost)                         │
│   Read           ████████░  $14.20  (1,204×)│
│   mcp_playwright █████░░░░   $8.40  (  311×)│
│   Bash           ██░░░░░░░   $3.10  (  192×)│
│   ...                                    │
├──────────────────────────────────────────┤
│  Open data folder    Quit                │
└──────────────────────────────────────────┘

Latency tab body:
├──────────────────────────────────────────┤
│  Tool latency (avg per call)             │
│   mcp_playwright ████████░  2.40s  (  311×)│
│   Bash           █████░░░░  1.60s  (  192×)│
│   Read           ██░░░░░░░   240 ms (1,204×)│
│   ...                                    │
```

**Refresh:** FSEvents watcher on `~/.cc-token-bar/sessions/` and `tools/`. Recompute aggregates on change. Plus a 5 s repeating timer that runs only while the popover is shown (started in `popoverDidShow`, invalidated in `popoverDidClose`). A one-shot refresh fires on every click-to-open so the panel never paints stale data.

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
| 7 | Panel tabs             | Two tabs: **Cost** (default, the v1 metric set) and **Latency** (avg latency per tool). Shared header/footer |
| 8 | Playwright tools       | Collapse all `mcp__playwright__*` to one `mcp_playwright` row in both tabs                                 |
| 9 | Latency source         | Transcript `tool_use`→`tool_result` timestamp delta; no hook changes                                      |

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
