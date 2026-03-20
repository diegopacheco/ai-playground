# Truth Detector

Analyzes GitHub users' recent public commit activity using an LLM (via CLI) to classify commit depth and produce summaries. For each of the last 10 public commits, it returns a classification (**DEEP**, **DECENT**, or **SHALLOW**), a one-line summary, and a numeric score (1-10). Scores are aggregated weekly to track trends over time.

## How It Works

```
GitHub Profile URL
       |
       v
 Fetch last 10 commits (GitHub Events API + Commit Details API)
       |
       v
 Single batched LLM call (claude CLI) analyzes all 10 commits at once
       |
       v
 Classification (DEEP/DECENT/SHALLOW) + Summary + Score per commit
       |
       v
 Store results in SQLite, aggregate weekly scores
```

### Classification Criteria

| Classification | Meaning | Score |
|---------------|---------|-------|
| **DEEP** | Substantial architectural change, complex logic, significant feature, non-trivial bug fix | 7-10 |
| **DECENT** | Meaningful contribution, moderate complexity, clear purpose | 4-6 |
| **SHALLOW** | Config tweak, version bump, typo fix, auto-generated, trivial rename | 1-3 |

## Tech Stack

### Backend
- **Rust 1.94+** (Edition 2024)
- **Actix-Web 4** + **Tokio** async runtime
- **SQLite** (WAL mode) via rusqlite
- **reqwest** for GitHub API calls
- **SSE** (Server-Sent Events) for real-time progress streaming
- LLM invocation via CLI subprocess (claude, gemini, or copilot)

### Frontend
- **Deno 2.x** runtime
- **React 19** + **TypeScript**
- **TanStack Router** (file-based routing)
- **TanStack Query** (server state management)
- **TanStack Table** (sortable leaderboard tables)
- **TailwindCSS 4.x** (dark theme)

## Project Structure

```
truth-detector/
├── run.sh                         # Start both backend + frontend
├── stop.sh                        # Stop both
├── design-doc.md                  # Full design specification
├── backend/
│   ├── Cargo.toml
│   ├── run.sh / stop.sh
│   └── src/
│       ├── main.rs                # Actix-web server, CORS, port 3000
│       ├── router.rs              # All route definitions
│       ├── models/types.rs        # Shared structs, SSE event types
│       ├── persistence/db.rs      # SQLite schema + CRUD
│       ├── sse/broadcaster.rs     # SSE channel management
│       ├── github/
│       │   ├── client.rs          # GitHub API: events + commit details
│       │   ├── cache.rs           # 1-hour SQLite cache for GitHub data
│       │   └── types.rs           # GitHub API response structs
│       ├── llm/
│       │   ├── prompt.rs          # Batched prompt for all 10 commits
│       │   ├── runner.rs          # CLI subprocess (30s timeout)
│       │   └── parser.rs          # Parse JSON response + fallback
│       ├── engine/
│       │   ├── analyzer.rs        # Orchestration: fetch -> LLM -> store
│       │   └── scoring.rs         # Score calculation, weekly aggregation
│       └── handlers/
│           ├── analysis.rs        # POST /analyze, GET /analyze/:id
│           ├── stream.rs          # SSE streaming endpoint
│           └── aggregation.rs     # Leaderboard + weekly scores
└── frontend/
    ├── deno.json / package.json
    ├── run.sh / stop.sh
    └── src/
        ├── main.tsx / App.tsx
        ├── api/client.ts          # Fetch wrappers for all endpoints
        ├── hooks/                 # useAnalysis, useAnalysisSSE, useLeaderboard
        ├── routes/
        │   ├── __root.tsx         # Dark layout + tab navigation
        │   ├── index.tsx          # Tab 1: Analyze a single user
        │   └── leaderboard.tsx    # Tab 2: Weekly leaderboard ranking
        └── components/
            ├── UserInput.tsx
            ├── AnalysisProgress.tsx
            ├── CommitCard.tsx
            ├── ScoreSummary.tsx
            ├── WeeklyChart.tsx
            ├── LeaderboardForm.tsx
            ├── LeaderboardTable.tsx
            ├── RankBadge.tsx
            └── ClassificationBadge.tsx
```

## UI

### Tab 1 - Analyze User
Enter a GitHub profile URL or username. The tool fetches the last 10 public commits, sends them to an LLM in a single batched call, and displays:
- Per-commit classification badge (DEEP/DECENT/SHALLOW), one-line summary, and score
- Total score out of 100, average score, classification breakdown
- Weekly score trend chart

### Tab 2 - Leaderboard
Add multiple GitHub users to a tracking pool. Users are ranked by weekly average score.
"Who wrote the deepest code this week?"

## Prerequisites

- Rust 1.94+ (`rustup update`)
- Deno 2.x (`brew install deno` or https://deno.com)
- Claude CLI installed and authenticated (`claude` command available in PATH)

## Running

```bash
./run.sh
```

Starts both:
- Backend on http://localhost:3000
- Frontend on http://localhost:5173

```bash
./stop.sh
```

Stops both services and clears ports.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_CLI` | `claude` | CLI to use: claude, gemini, copilot |
| `LLM_MODEL` | `sonnet` | Model to pass to the CLI |
| `LLM_TIMEOUT` | `30` | Seconds before CLI timeout |
| `BACKEND_PORT` | `3000` | Backend listen port |
| `FRONTEND_PORT` | `5173` | Frontend dev server port |
| `DATABASE_PATH` | `./truth_detector.db` | SQLite file path |

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/analyze` | Start analysis for a GitHub user |
| GET | `/api/analyze/:id` | Get full analysis with commit results |
| GET | `/api/analyze/:id/stream` | SSE stream of analysis progress |
| GET | `/api/users/:username/weekly` | Weekly score history |
| GET | `/api/users/:username/latest` | Most recent analysis |
| GET | `/api/leaderboard` | Current week leaderboard |
| POST | `/api/leaderboard/track` | Add user to leaderboard |
| DELETE | `/api/leaderboard/track/:username` | Remove user from leaderboard |

## GitHub API & Caching

Uses the public GitHub Events API (unauthenticated, 60 requests/hour). Fetched commits are cached in SQLite with a 1-hour TTL, so repeated analysis of the same user within an hour costs zero GitHub API calls.
