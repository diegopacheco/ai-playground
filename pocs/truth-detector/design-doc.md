# Truth Detector - Design Document

## Overview

Truth Detector is a tool that analyzes GitHub users' recent public commit activity using an LLM (via CLI invocation) to classify commit depth and produce summaries. It fetches the last 10 public commits for a given GitHub profile, sends each commit diff/message to an LLM for analysis, and returns a classification (DEEP, DECENT, or SHALLOW), a one-line summary, and a numeric score. Scores are aggregated weekly to track trends over time.

## Architecture

```
┌─────────────────────────────────────┐
│           Frontend (Deno)           │
│  React 19 + TanStack + TailwindCSS │
│         localhost:5173              │
└──────────────┬──────────────────────┘
               │ HTTP / SSE
┌──────────────▼──────────────────────┐
│        Backend (Rust 1.94+)         │
│      Actix-Web + Tokio + SQLite     │
│         localhost:3000              │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │ GitHub API │  │ LLM (CLI)     │  │
│  │  Client    │  │ claude/gemini │  │
│  └───────────┘  └────────────────┘  │
└─────────────────────────────────────┘
```

## LLM Integration

Following the same pattern as `agents-auction-hourse`, the LLM is invoked via CLI subprocess. The backend spawns a process (e.g., `claude -p "<prompt>" --model sonnet --dangerously-skip-permissions`) and parses the JSON response.

### Batched Prompt Strategy

All 10 commits are sent in a single LLM call to minimize latency (~15-20s total instead of ~100-200s). The prompt contains all commits bundled together with their diffs (each diff truncated to 2000 chars to fit within context).

The LLM receives a single prompt with:
- All 10 commits listed as `[1]` through `[10]`
- Each entry contains: repo name, commit SHA, commit message, diff (truncated to 2000 chars)
- Author info

The LLM must respond with a JSON array:
```json
{
  "results": [
    {
      "index": 1,
      "classification": "DEEP" | "DECENT" | "SHALLOW",
      "summary": "one-line summary of the commit",
      "score": 1-10
    },
    ...
  ]
}
```

If the batched call fails or times out (30s), the entire batch falls back to defaults (all DECENT, score 5, first 80 chars of each commit message as summary).

### Classification Criteria

| Classification | Meaning | Score Range |
|---------------|---------|-------------|
| **DEEP** | Substantial architectural change, complex logic, significant feature, non-trivial bug fix | 7-10 |
| **DECENT** | Meaningful contribution, moderate complexity, clear purpose | 4-6 |
| **SHALLOW** | Config tweak, version bump, typo fix, auto-generated, trivial rename | 1-3 |

### Fallback

If the batched LLM CLI call fails or times out (30s), all commits in the batch get:
- Classification: `DECENT` (neutral default)
- Summary: first 80 chars of the commit message
- Score: 5
- Marked with `fallback: true`

If the LLM returns partial results (fewer than 10), missing commits use the same fallback values.

## Backend (Rust 1.94+, Edition 2024)

### Crate Dependencies

| Crate | Purpose |
|-------|---------|
| `actix-web` | HTTP server, routing, CORS |
| `actix-cors` | CORS middleware |
| `tokio` | Async runtime |
| `rusqlite` | SQLite with bundled feature |
| `serde` / `serde_json` | JSON serialization |
| `reqwest` | GitHub API HTTP client |
| `uuid` | Request/analysis IDs |
| `chrono` | Timestamps, weekly aggregation |

### Module Structure

```
backend/
├── Cargo.toml
├── src/
│   ├── main.rs              (server setup, actix-web config, CORS)
│   ├── router.rs            (all route definitions)
│   ├── handlers/
│   │   ├── mod.rs
│   │   ├── analysis.rs      (analyze user, get analysis by id)
│   │   ├── stream.rs        (SSE streaming analysis progress)
│   │   └── aggregation.rs   (weekly scores, multi-user comparison)
│   ├── github/
│   │   ├── mod.rs
│   │   ├── client.rs        (fetch commits via GitHub REST API)
│   │   ├── cache.rs         (SQLite-backed cache, 1-hour TTL, cache-or-fetch logic)
│   │   └── types.rs         (GitHub API response structs)
│   ├── llm/
│   │   ├── mod.rs
│   │   ├── runner.rs        (CLI subprocess execution, 30s timeout)
│   │   ├── prompt.rs        (batched prompt building for all 10 commits)
│   │   └── parser.rs        (parse batched LLM JSON array response, fallback)
│   ├── engine/
│   │   ├── mod.rs
│   │   ├── analyzer.rs      (orchestrates: fetch commits -> LLM -> store)
│   │   └── scoring.rs       (score calculation, weekly aggregation logic)
│   ├── persistence/
│   │   ├── mod.rs
│   │   └── db.rs            (SQLite schema, CRUD operations)
│   ├── sse/
│   │   ├── mod.rs
│   │   └── broadcaster.rs   (SSE event broadcasting to frontend)
│   └── models/
│       ├── mod.rs
│       └── types.rs         (shared data structures, SSE event types)
```

### Database Schema (SQLite with WAL mode)

```sql
analyses
  id          TEXT PRIMARY KEY,
  github_user TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  total_score INTEGER,
  avg_score   REAL,
  status      TEXT NOT NULL  -- "running", "completed", "failed"

commits
  id              TEXT PRIMARY KEY,
  analysis_id     TEXT NOT NULL REFERENCES analyses(id),
  repo_name       TEXT NOT NULL,
  commit_sha      TEXT NOT NULL,
  commit_message  TEXT NOT NULL,
  commit_date     TEXT NOT NULL,
  classification  TEXT NOT NULL,  -- DEEP, DECENT, SHALLOW
  summary         TEXT NOT NULL,
  score           INTEGER NOT NULL,
  fallback        INTEGER NOT NULL DEFAULT 0,
  raw_llm_output  TEXT

tracked_users
  github_user TEXT PRIMARY KEY,
  added_at    TEXT NOT NULL

github_cache
  id          TEXT PRIMARY KEY,
  github_user TEXT NOT NULL,
  payload     TEXT NOT NULL,       -- JSON array of fetched commits with diffs
  fetched_at  TEXT NOT NULL,       -- ISO timestamp
  expires_at  TEXT NOT NULL,       -- fetched_at + 1 hour
  UNIQUE(github_user)

weekly_scores
  id          TEXT PRIMARY KEY,
  github_user TEXT NOT NULL,
  week_start  TEXT NOT NULL,  -- ISO date of Monday
  week_end    TEXT NOT NULL,  -- ISO date of Sunday
  total_score INTEGER NOT NULL,
  avg_score   REAL NOT NULL,
  num_commits INTEGER NOT NULL,
  deep_count  INTEGER NOT NULL,
  decent_count INTEGER NOT NULL,
  shallow_count INTEGER NOT NULL,
  UNIQUE(github_user, week_start)
```

### API Endpoints

| Method | Endpoint | Handler | Returns |
|--------|----------|---------|---------|
| POST | `/api/analyze` | `analysis::create` | `{"id": "uuid"}` starts async analysis |
| GET | `/api/analyze/:id` | `analysis::get` | Full analysis with all commit results |
| GET | `/api/analyze/:id/stream` | `stream::sse` | SSE stream of analysis progress |
| GET | `/api/users/:username/weekly` | `aggregation::weekly` | Weekly score history for a user |
| GET | `/api/leaderboard` | `aggregation::leaderboard` | All tracked users ranked by current week avg score |
| GET | `/api/leaderboard/week/:week_start` | `aggregation::leaderboard_week` | Leaderboard for a specific past week |
| POST | `/api/leaderboard/track` | `aggregation::track` | Add a user to the leaderboard tracking pool |
| DELETE | `/api/leaderboard/track/:username` | `aggregation::untrack` | Remove a user from leaderboard tracking |
| GET | `/api/users/:username/latest` | `analysis::latest` | Most recent analysis for a user |

### SSE Events

| Event | Payload | When |
|-------|---------|------|
| `analysis_start` | `{github_user, commit_count, cached}` | Analysis begins, `cached` indicates GitHub cache hit |
| `commits_fetched` | `[{sha, message, repo}, ...]` | All commits fetched (from API or cache) |
| `llm_thinking` | `{}` | Batched LLM call in progress |
| `all_commits_analyzed` | `[{index, sha, classification, summary, score, fallback}, ...]` | All 10 results ready at once |
| `analysis_complete` | `{total_score, avg_score, deep, decent, shallow}` | Scores calculated, stored |
| `error` | `{message}` | Error occurred |

### GitHub API Integration

- Endpoint: `https://api.github.com/users/{username}/events?per_page=30`
- Filter for `PushEvent` type, extract commits
- Take the first 10 unique commits across repos
- For each commit, fetch the diff via `https://api.github.com/repos/{owner}/{repo}/commits/{sha}`
- Rate limit: 60 req/hr (unauthenticated, no token needed)
- Plain HTTP GET requests via reqwest, no authentication required

### GitHub Response Cache

Fetched commits (events + diffs) are cached in the `github_cache` table with a 1-hour TTL. On each analysis request:
1. Check `github_cache` for the user where `expires_at > now()`
2. If cache hit: deserialize the stored JSON payload, skip all GitHub API calls
3. If cache miss or expired: fetch fresh data from GitHub, store in cache (UPSERT on github_user)
4. This reduces GitHub API usage from ~11 requests per analysis to 0 for repeated lookups within the hour

### Analysis Flow

```
1. POST /api/analyze {github_user: "diegopacheco"}
2. Return {id: "uuid"} immediately
3. Async (tokio::spawn):
   a. Check github_cache for fresh data (TTL 1 hour)
   b. If cache miss: fetch last 10 commits from GitHub Events API + diffs, store in cache
   c. Broadcast SSE: analysis_start
   d. Build single batched prompt with all 10 commits
   e. Spawn one CLI process (claude -p "..." --model sonnet --dangerously-skip-permissions)
   f. Parse batched JSON array response, apply fallback for missing/failed entries
   g. Store all commit results in SQLite
   h. Broadcast SSE: all_commits_analyzed (with full results array)
   i. Calculate total/avg scores
   j. Upsert weekly_scores row
   k. Broadcast SSE: analysis_complete
```

## Frontend (Deno + React 19 + TanStack + TailwindCSS)

### Tech Stack

| Tech | Version | Purpose |
|------|---------|---------|
| Deno | 2.x | Runtime, package management, dev server |
| React | 19 | UI library |
| TanStack Router | latest | File-based routing, type-safe navigation |
| TanStack Query | latest | Server state, caching, refetching |
| TanStack Table | latest | Sortable tables for aggregated results |
| TailwindCSS | 4.x | Utility-first styling |
| Vite | 6.x | Build tool (via Deno) |

### Module Structure

```
frontend/
├── deno.json
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
├── src/
│   ├── main.tsx                (entry point, QueryClientProvider)
│   ├── App.tsx                 (RouterProvider setup)
│   ├── types/
│   │   └── index.ts            (all TypeScript interfaces)
│   ├── api/
│   │   └── client.ts           (fetch wrappers for all endpoints)
│   ├── hooks/
│   │   ├── useAnalysisSSE.ts   (SSE event listener for live progress)
│   │   ├── useAnalysis.ts      (React Query hooks for analysis CRUD)
│   │   └── useLeaderboard.ts   (React Query hooks for leaderboard/ranking)
│   ├── routes/
│   │   ├── __root.tsx          (root layout, tab navigation)
│   │   ├── index.tsx           (Tab 1 - single user analysis)
│   │   └── leaderboard.tsx     (Tab 2 - weekly leaderboard ranking)
│   └── components/
│       ├── UserInput.tsx        (GitHub profile URL input + analyze button)
│       ├── AnalysisProgress.tsx (live SSE-driven progress indicator)
│       ├── CommitCard.tsx       (single commit: classification badge, summary, score)
│       ├── ScoreSummary.tsx     (total/avg score, classification breakdown pie)
│       ├── WeeklyChart.tsx      (weekly score trend line)
│       ├── LeaderboardForm.tsx  (add users to the leaderboard pool)
│       ├── LeaderboardTable.tsx (TanStack Table: ranked users by weekly avg score)
│       ├── RankBadge.tsx        (1st/2nd/3rd place styling)
│       └── ClassificationBadge.tsx (colored DEEP/DECENT/SHALLOW badge)
```

### UI Design

#### Tab 1 - Single User Analysis

```
┌─────────────────────────────────────────────────┐
│  [Tab 1: Analyze User]  [Tab 2: Compare Users]  │
├─────────────────────────────────────────────────┤
│                                                  │
│  GitHub Profile URL:                             │
│  ┌────────────────────────────────┐ [Analyze]    │
│  │ https://github.com/username   │              │
│  └────────────────────────────────┘              │
│                                                  │
│  ── Analysis Progress ──────────────────         │
│  [████████░░] 8/10 commits analyzed              │
│                                                  │
│  ── Score Summary ──────────────────────         │
│  Total: 62/100  |  Avg: 6.2  |  DEEP x3         │
│  DECENT x5  |  SHALLOW x2                        │
│                                                  │
│  ── Commit Results ─────────────────────         │
│  ┌─────────────────────────────────────┐         │
│  │ [DEEP]  repo/project  abc1234       │         │
│  │ Refactored auth middleware to use    │         │
│  │ token-based validation               │         │
│  │ Score: 8/10                          │         │
│  └─────────────────────────────────────┘         │
│  ┌─────────────────────────────────────┐         │
│  │ [SHALLOW]  repo/docs  def5678       │         │
│  │ Fixed typo in README                 │         │
│  │ Score: 2/10                          │         │
│  └─────────────────────────────────────┘         │
│  ...                                             │
│                                                  │
│  ── Weekly Trend ───────────────────────         │
│  Score                                           │
│  80 |        *                                   │
│  60 |  *  *     *                                │
│  40 |              *                             │
│     └──────────────────── Weeks                  │
│                                                  │
└─────────────────────────────────────────────────┘
```

#### Tab 2 - Weekly Leaderboard

Users added to the leaderboard pool are ranked by weekly average score. The leaderboard answers: "Who wrote the deepest code this week?"

```
┌──────────────────────────────────────────────────────┐
│  [Tab 1: Analyze User]  [Tab 2: Leaderboard]         │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Add user to leaderboard:                             │
│  ┌────────────────────┐ [Add & Analyze]               │
│  │ github username     │                              │
│  └────────────────────┘                               │
│  Tracking: [diegopacheco x] [torvalds x] [graydon x] │
│                                                       │
│  ── This Week (Mar 16 - Mar 22) ──────────────        │
│  ┌────┬──────────────┬─────┬───────┬────┬────┬────┐  │
│  │ #  │ User         │ Avg │ Total │ D  │ De │ Sh │  │
│  ├────┼──────────────┼─────┼───────┼────┼────┼────┤  │
│  │ 1  │ torvalds     │ 8.1 │  81   │ 7  │ 2  │ 1  │  │
│  │ 2  │ diegopacheco │ 7.2 │  72   │ 5  │ 3  │ 2  │  │
│  │ 3  │ graydon      │ 6.5 │  65   │ 3  │ 5  │ 2  │  │
│  └────┴──────────────┴─────┴───────┴────┴────┴────┘  │
│  (sortable columns, ranked by Avg descending)         │
│                                                       │
│  ── Previous Weeks ───────────────────────────        │
│  [v Mar 09 - Mar 15]                                  │
│  ┌────┬──────────────┬─────┬───────┬────┬────┬────┐  │
│  │ 1  │ diegopacheco │ 5.8 │  58   │ 2  │ 6  │ 2  │  │
│  │ 2  │ ...          │     │       │    │    │    │  │
│  └────┴──────────────┴─────┴───────┴────┴────┴────┘  │
│  (collapsible per week)                               │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Classification Badge Colors

| Classification | Color | TailwindCSS |
|---------------|-------|-------------|
| DEEP | Green | `bg-green-500/20 text-green-400 border-green-500/30` |
| DECENT | Yellow | `bg-yellow-500/20 text-yellow-400 border-yellow-500/30` |
| SHALLOW | Red | `bg-red-500/20 text-red-400 border-red-500/30` |

## Scoring Algorithm

### Per-Commit Score
- LLM assigns 1-10 based on classification criteria
- Fallback commits always get 5

### Per-Analysis Score
- `total_score` = sum of all 10 commit scores (max 100)
- `avg_score` = total_score / num_commits

### Weekly Aggregation
- Groups all analyses for a user within Monday-Sunday window
- `weekly_total` = sum of all commit scores that week
- `weekly_avg` = weekly_total / total commits that week
- Classification counts tracked separately (deep_count, decent_count, shallow_count)
- Re-calculated on each new analysis for the current week

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_CLI` | `claude` | Which CLI to use (claude, gemini, copilot) |
| `LLM_MODEL` | `sonnet` | Model to pass to the CLI |
| `LLM_TIMEOUT` | `30` | Seconds before batched CLI timeout |
| `BACKEND_PORT` | `3000` | Actix-web listen port |
| `FRONTEND_PORT` | `5173` | Vite dev server port |
| `DATABASE_PATH` | `./truth_detector.db` | SQLite file path |

## Scripts

### run.sh
- Builds and starts the Rust backend on port 3000
- Starts the Deno/Vite frontend on port 5173
- Saves PIDs to `/tmp/truth-detector-*.pid`

### stop.sh
- Kills backend and frontend by PID files
- Kills any remaining processes on ports 3000 and 5173
- Cleans up PID files

## Constraints and Considerations

1. **GitHub Rate Limits**: Unauthenticated API at 60 requests/hour. Each fresh analysis uses ~11 requests (1 events + 10 commit details). The 1-hour cache means repeated lookups for the same user cost 0 requests.
2. **LLM Latency**: Single batched call for all 10 commits takes ~15-20s via CLI. SSE streaming shows phases (fetching, thinking, done).
3. **SQLite Concurrency**: WAL mode enables concurrent reads while one writer operates. Sufficient for this use case.
4. **Commit Diff Size**: Large diffs are truncated to 2000 chars per commit (10 commits x 2000 = 20000 chars max in the batched prompt).
5. **Rust Edition 2024**: Requires Rust 1.94+. Uses async/await throughout with Tokio runtime.
