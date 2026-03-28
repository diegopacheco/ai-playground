# Agent PR Monitor - Design Doc

## Overview

A Rust CLI tool that monitors GitHub Pull Requests using LLM agents.
The user runs `agent-pr` in the terminal, pastes a GitHub PR link, and the tool clones the repo,
checks out the PR branch, and monitors it every 5 minutes.
It automatically fixes compilation issues, adds missing tests, and responds to PR comments.

## Tech Stack

- **Language**: Rust 1.94, Edition 2024
- **LLM Integration**: CLI-based subprocess invocation (same pattern as agents-auction-hourse)
  - `claude -p <prompt> --model <model> --dangerously-skip-permissions`
  - `gemini -y -p <prompt>`
  - `copilot --allow-all --model <model> -p <prompt>`
  - `codex exec --full-auto -m <model> <prompt>`
- **GitHub CLI**: `gh` for PR operations (clone, comments, status)
- **No external Rust crates for LLM** - just `std::process::Command`
- **Web Server**: Actix-Web + Tokio (async runtime) for serving the dashboard
- **Frontend**: React 19, TypeScript, Vite, Bun
- **Frontend Embedding**: The built frontend is embedded into the Rust binary using `rust-embed` so the dashboard ships as a single binary

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          agent-pr CLI                                │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐               │
│  │  CLI     │──▶│  PR Manager  │──▶│  Monitor Loop  │               │
│  │  Input   │   │  (gh clone)  │   │  (5 min cycle) │               │
│  └──────────┘   └──────────────┘   └───────┬────────┘               │
│                                            │                         │
│       --ui flag?                           │                         │
│       ┌──────────────────────┐             │                         │
│       │  Actix-Web Server    │◀────────────┤                         │
│       │  :3000               │   shared    │                         │
│       │  ┌────────────────┐  │   state     │                         │
│       │  │ Embedded React │  │  (Arc<M>)   │                         │
│       │  │ Dashboard SPA  │  │             │                         │
│       │  └────────────────┘  │             │                         │
│       │  REST API + SSE      │             │                         │
│       └──────────────────────┘             │                         │
│                                            │                         │
│                    ┌───────────────────────┬┴──────┐                 │
│                    ▼                       ▼       ▼                 │
│            ┌──────────────┐  ┌──────────┐  ┌────────┐               │
│            │  Compile     │  │  Test    │  │Comment │               │
│            │  Fixer Agent │  │  Agent   │  │Agent   │               │
│            └──────┬───────┘  └────┬─────┘  └───┬────┘               │
│                   │               │             │                    │
│                   ▼               ▼             ▼                    │
│            ┌──────────────────────────────────────────┐              │
│            │         LLM Runner (CLI subprocess)      │              │
│            │   claude | gemini | copilot | codex      │              │
│            └──────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
   /tmp/agent-pr/         git commit          gh pr comment
   <repo-clone>           + git push          (reply)
```

## CLI Flags

```
agent-pr [OPTIONS] <PR_URL>

OPTIONS:
  --llm <agent>        LLM agent to use: claude, gemini, copilot, codex (required)
  --model <model>      Model for the LLM agent (required)
  --refresh <duration> Refresh interval: 1s, 30s, 1m, 5m, 10m etc. (default: 5m)
  --dry-run            Download PR and apply fixes locally but never push or comment on GitHub
  --ui                 Open the web dashboard
  --port <port>        Dashboard port (default: 3000, only with --ui)
```

Duration format: a number followed by `s` (seconds) or `m` (minutes). Examples: `1s`, `30s`, `1m`, `5m`, `10m`.

## CLI Usage

### Headless Mode (default)
```
$ agent-pr --llm claude --model sonnet https://github.com/owner/repo/pull/123

Cloning PR #123 from owner/repo...
Monitoring started. Checking every 5 minutes. Press Ctrl+C to stop.

[12:00:00] Checking PR #123...
[12:00:05] No issues found. All good.
[12:05:00] Checking PR #123...
[12:05:10] Compilation error detected. Fixing...
[12:05:30] Fix committed and pushed.
[12:10:00] Checking PR #123...
[12:10:15] New comment from @reviewer: "Can you add tests for the edge case?"
[12:10:45] Replied to comment and added tests.
```

### Dashboard Mode
```
$ agent-pr --llm claude --model opus --ui https://github.com/owner/repo/pull/123

Cloning PR #123 from owner/repo...
Dashboard running at http://localhost:3000
Monitoring started. Checking every 5 minutes. Press Ctrl+C to stop.
```

### Custom Refresh Rate
```
$ agent-pr --llm claude --model sonnet --refresh 1m https://github.com/owner/repo/pull/123
$ agent-pr --llm gemini --model gemini-2.5-pro --refresh 30s --ui --port 8080 https://github.com/owner/repo/pull/123
```

### Dry Run Mode
```
$ agent-pr --llm claude --model sonnet --dry-run --ui https://github.com/owner/repo/pull/123
```

Downloads the PR, applies fixes locally, shows everything on the UI, but never does git commit, git push, or gh pr comment. Safe for testing.

No interactive prompts. All configuration via flags. PR URL is a positional argument.

## Core Modules

### 1. `main.rs` - CLI Entry Point
- Parses CLI flags: `--llm`, `--model`, `--refresh`, `--ui`, `--port`, and positional PR URL
- Validates that --llm and --model are provided, PR URL is valid
- Parses --refresh duration string (e.g. "1s", "30s", "1m", "5m") into seconds, default 300s (5m)
- If `--ui` flag is present, starts Actix-Web server (default port 3000, `--port` to override) and opens browser
- Starts the monitor loop with the configured refresh interval

### 2. `pr.rs` - PR Manager
- Uses `gh pr checkout` to clone the PR into `/tmp/agent-pr/<owner>-<repo>-<pr_number>/`
- Runs `git pull` to fetch latest changes
- Gets PR diff via `gh pr diff`
- Gets PR comments via `gh pr view --comments`
- Pushes commits via `git push`

### 3. `monitor.rs` - Monitor Loop
- Runs at the configured refresh interval (default 5m, configurable via --refresh flag)
- Every action is recorded into shared `AppState` so the dashboard can display it via REST/SSE
- Each cycle:
  1. `git pull` to get latest changes
  2. Run compilation check (`cargo build` for Rust, `go build` for Go, `npm run build` for JS/TS)
  3. Run test check (`cargo test`, `go test`, `npm test`)
  4. Check for new/unanswered PR comments via `gh api`
  5. Dispatch to the appropriate agent based on findings

### 4. `agents.rs` - LLM Agent Runner
- Generic runner that executes CLI commands with a timeout (30 seconds)
- Agent-specific builders for claude, gemini, copilot, codex
- Prompt construction with full context (error output, file contents, comment text)
- Response parsing (expects code blocks or plain text responses)

### 5. `compiler.rs` - Compilation Fixer Agent
- Detects project type by looking for Cargo.toml, go.mod, package.json
- Runs the build command and captures stderr
- If build fails, sends error + relevant source files to LLM
- Applies the fix, commits, and pushes
- Retries up to 10 times per issue, after that gives up and logs the reason why it failed

### 6. `tester.rs` - Test Agent
- Runs the test command for the detected project type
- If tests fail, sends failure output + source to LLM for fix
- LLM reads all code and the PR modified files, identifies test gaps, and writes missing tests
- Retries up to 10 times per issue, after that gives up and logs the reason why it failed
- Commits and pushes test additions/fixes

### 7. `commenter.rs` - Comment Agent
- Fetches PR comments via `gh api repos/{owner}/{repo}/pulls/{pr_number}/comments`
- Tracks which comments have been answered (by comment ID)
- For new comments, sends the comment + relevant code context to LLM
- Posts reply via `gh pr comment`

### 8. `state.rs` - Shared Application State
- Defines `AppState`, `AgentAction`, `Counters`, `CommentThread`, `FileEntry` structs
- All fields are `serde::Serialize` for JSON API responses
- State is wrapped in `Arc<Mutex<AppState>>` shared between monitor loop and web server
- Provides methods to record actions, increment counters, and refresh the file tree

### 9. `server.rs` - Actix-Web Dashboard Server
- Starts Actix-Web on port 3000 with tokio runtime
- Serves embedded React SPA via rust-embed for `/` and `/assets/*`
- REST API endpoints under `/api/*` reading from shared `AppState`
- SSE endpoint at `/api/events` that streams real-time updates
- SPA fallback: all non-API routes return `index.html`

## Dashboard (--ui mode)

### Overview

When `agent-pr --ui` is used, the Rust binary starts an Actix-Web server on port 3000
and serves an embedded React SPA. The frontend is built with Bun + Vite at compile time
and embedded into the binary using `rust-embed`. The dashboard auto-opens in the default browser.

### Shared State

The monitor loop and the web server share state via `Arc<Mutex<AppState>>`:

```rust
struct AppState {
    pr_info: PrInfo,
    actions: Vec<AgentAction>,
    counters: Counters,
    comments: Vec<CommentThread>,
    file_tree: Vec<FileEntry>,
    logs: Vec<AgentLog>,
}

struct PrInfo {
    url: String,
    owner: String,
    repo: String,
    pr_number: u64,
    title: String,
    branch: String,
    total_files: usize,
    clone_path: String,
}

struct AgentAction {
    id: u64,
    timestamp: String,
    action_type: ActionType,   // CompileFix | TestFix | TestAdd | CommentReply
    description: String,
    files_changed: Vec<String>,
    llm_agent: String,
    llm_model: String,
    commit_sha: Option<String>,
}

struct Counters {
    compilation_fixes: u64,
    test_fixes: u64,
    comments_answered: u64,
    total_cycles: u64,
}

struct CommentThread {
    id: u64,
    github_comment_id: u64,
    author: String,
    body: String,
    file_path: Option<String>,
    line: Option<u64>,
    timestamp: String,
    replies: Vec<CommentReply>,
}

struct CommentReply {
    author: String,
    body: String,
    timestamp: String,
    is_agent: bool,
}

struct FileEntry {
    path: String,
    name: String,
    is_dir: bool,
    children: Vec<FileEntry>,
}

struct AgentLog {
    id: u64,
    timestamp: String,
    action_type: ActionType,
    llm_agent: String,
    llm_model: String,
    prompt: String,
    response: String,
    result: String,
    commit_sha: Option<String>,
}
```

### REST API Endpoints (Actix-Web)

| Method | Endpoint                | Description                                    |
|--------|-------------------------|------------------------------------------------|
| GET    | `/`                     | Serves the embedded React SPA (index.html)     |
| GET    | `/assets/*`             | Serves embedded static assets (JS, CSS)        |
| GET    | `/api/status`           | Returns PrInfo + Counters                      |
| GET    | `/api/actions`          | Returns all AgentAction entries                 |
| GET    | `/api/comments`         | Returns all CommentThread with replies          |
| GET    | `/api/files`            | Returns file tree of the cloned repo            |
| GET    | `/api/files/content`    | Returns file content by path (`?path=src/main.rs`) |
| GET    | `/api/logs`             | Returns all AgentLog entries (prompts+responses) |
| GET    | `/api/health`           | Returns uptime, cycle count, last check time     |
| GET    | `/api/events`           | SSE stream for real-time updates                 |

### SSE Events

The dashboard uses Server-Sent Events to get real-time updates without polling:

| Event              | Payload                     | When                                  |
|--------------------|-----------------------------|---------------------------------------|
| `cycle_start`      | `{ cycle: number }`         | Monitor cycle begins                  |
| `cycle_end`        | `{ cycle, status }`         | Monitor cycle completes               |
| `action`           | `AgentAction`               | Agent performs any action              |
| `counter_update`   | `Counters`                  | Any counter changes                   |
| `new_comment`      | `CommentThread`             | New PR comment detected               |
| `comment_reply`    | `{ thread_id, reply }`      | Agent replies to a comment            |

### Frontend Architecture

**Built with**: React 19, TypeScript, Vite, Bun

**Build process**: `bun install && bun run build` produces static files in `frontend/dist/`.
The Rust build script (`build.rs`) does NOT run bun. The frontend must be pre-built before `cargo build`.
`rust-embed` includes `frontend/dist/` into the binary at compile time.

#### Three Tabs Layout

**Tab 1: Monitor Dashboard** (default)

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent PR Monitor                              [Tab1] [Tab2] [Tab3]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PR: owner/repo#123 - "Fix auth middleware"     [View on GH ↗] │
│  Branch: fix/auth-middleware                                    │
│  Files: 12 | Agent: claude (sonnet)                             │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐        │
│  │ Compile Fixes│ │  Test Fixes  │ │ Comments Answered │        │
│  │      3       │ │      5       │ │        7          │        │
│  └──────────────┘ └──────────────┘ └──────────────────┘        │
│                                                                 │
│  ┌─ Activity Log ───────────────────────────────────────────┐   │
│  │ [12:30:00] Compilation fix: fixed missing import in      │   │
│  │            src/auth.rs (commit a1b2c3d)                   │   │
│  │ [12:25:00] Test added: added 3 tests for UserService     │   │
│  │ [12:20:00] Comment reply: answered @reviewer question    │   │
│  │ [12:15:00] No issues found. All good.                    │   │
│  │ [12:10:00] Test fix: fixed assertion in auth_test.rs     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ File Explorer ──────────────┐ ┌─ File Viewer ───────────┐  │
│  │ 📁 src/                      │ │ src/auth.rs             │  │
│  │   ├── main.rs                │ │                         │  │
│  │   ├── auth.rs           ◀──  │ │  1│ use std::collections│  │
│  │   ├── models/                │ │  2│ use crate::db;      │  │
│  │   │   └── user.rs            │ │  3│                     │  │
│  │   └── tests/                 │ │  4│ pub fn validate(    │  │
│  │       └── auth_test.rs       │ │  5│     token: &str     │  │
│  │ 📁 Cargo.toml                │ │  6│ ) -> bool {         │  │
│  └──────────────────────────────┘ │  7│     ...             │  │
│                                   └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Tab 2: Comments & Threads**

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent PR Monitor                              [Tab1] [Tab2] [Tab3]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PR Comments (7 threads, 4 answered by agent)                   │
│                                                                 │
│  ┌─ Thread #1 ── src/auth.rs:42 ───────────────────────────┐   │
│  │ @reviewer (human) - 12:05:00                             │   │
│  │ "This validation logic doesn't handle expired tokens.    │   │
│  │  Can you add that check?"                                │   │
│  │                                                          │   │
│  │   ↳ @agent-pr (agent) - 12:10:45                        │   │
│  │     "Added expiration check in validate(). The token     │   │
│  │      now checks the exp claim against current UTC time.  │   │
│  │      See commit a1b2c3d."                                │   │
│  │                                                          │   │
│  │   ↳ @reviewer (human) - 12:15:00                        │   │
│  │     "Looks good, thanks!"                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Thread #2 ── src/models/user.rs:18 ────────────────────┐   │
│  │ @maintainer (human) - 12:20:00                           │   │
│  │ "Why did you make this field public? It should be        │   │
│  │  behind a getter."                                       │   │
│  │                                                          │   │
│  │   ↳ @agent-pr (agent) - 12:25:30                        │   │
│  │     "Changed to private with a pub getter method.        │   │
│  │      See commit d4e5f6g."                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Tab 3: Agent Logs**

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent PR Monitor                       [Tab1] [Tab2] [Tab3]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Agent Logs (raw LLM prompts and responses)                     │
│                                                                 │
│  ┌─ Log #5 ── CompileFix ── 12:30:00 ── claude (sonnet) ───┐   │
│  │                                                          │   │
│  │  PROMPT:                                                 │   │
│  │  You are a senior developer. The following code has a    │   │
│  │  compilation error. Fix the error and return ONLY the    │   │
│  │  corrected file content.                                 │   │
│  │  File: src/auth.rs                                       │   │
│  │  Error: error[E0433]: failed to resolve: use of          │   │
│  │  undeclared crate `chrono`                               │   │
│  │  ...                                                     │   │
│  │                                                          │   │
│  │  RESPONSE:                                               │   │
│  │  ```rust                                                 │   │
│  │  use std::time::{SystemTime, UNIX_EPOCH};                │   │
│  │  pub fn validate(token: &str) -> bool {                  │   │
│  │      ...                                                 │   │
│  │  ```                                                     │   │
│  │                                                          │   │
│  │  RESULT: applied, commit a1b2c3d                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Log #4 ── TestAdd ── 12:25:00 ── claude (sonnet) ──────┐   │
│  │  PROMPT: ...                                             │   │
│  │  RESPONSE: ...                                           │   │
│  │  RESULT: applied, commit b2c3d4e                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### File Viewer with Syntax Highlighting

- Uses a lightweight syntax highlighting library (Prism.js or highlight.js) bundled in the frontend
- Supports Rust, Go, TypeScript, Java, JSON, YAML, TOML, Markdown
- Line numbers displayed on the left
- Color theme: dark theme matching GitHub dark mode
- Click a file in the explorer tree to view it
- File content fetched from `/api/files/content?path=<relative_path>`

#### Frontend File Structure

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts
│   ├── hooks/
│   │   └── useSSE.ts
│   ├── components/
│   │   ├── Header.tsx
│   │   ├── CounterCards.tsx
│   │   ├── ActivityLog.tsx
│   │   ├── FileExplorer.tsx
│   │   ├── FileViewer.tsx
│   │   ├── CommentThreads.tsx
│   │   ├── CommentThread.tsx
│   │   └── AgentLogs.tsx
│   └── types/
│       └── index.ts
└── dist/                  (build output, embedded into binary)
```

### Embedding Strategy

```
// In Rust, using rust-embed
#[derive(RustEmbed)]
#[folder = "frontend/dist/"]
struct FrontendAssets;
```

Actix-Web serves these embedded files:
- `GET /` returns `index.html`
- `GET /assets/*` returns JS/CSS bundles
- All other non-`/api/*` routes fallback to `index.html` (SPA routing)

## Project Detection

The monitor detects the project type by checking for:

| File           | Project Type | Build Command         | Test Command          |
|----------------|-------------|----------------------|----------------------|
| Cargo.toml     | Rust        | `cargo build`        | `cargo test`         |
| go.mod         | Go          | `go build ./...`     | `go test ./...`      |
| package.json   | Node/TS     | `npm run build`      | `npm test`           |
| pom.xml        | Java/Maven  | `mvn compile`        | `mvn test`           |
| build.gradle   | Java/Gradle | `gradle build`       | `gradle test`        |

## LLM Prompt Templates

### Compilation Fix Prompt
```
You are a senior developer. The following code has a compilation error.
Fix the error and return ONLY the corrected file content.

Project type: {project_type}
File: {file_path}
Error:
{compiler_error}

Current file content:
{file_content}
```

### Missing Test Prompt
```
You are a senior developer. Write tests for the following code.
Return ONLY the test file content.

Project type: {project_type}
Source file: {file_path}
Source content:
{file_content}

Existing test files in project:
{existing_test_files}
```

### Comment Reply Prompt
```
You are a developer working on this PR. Reply to the following review comment.
If the comment requests a code change, return the fix.
If it is a question, provide a clear answer.

PR Title: {pr_title}
PR Description: {pr_description}
Comment by @{author}: {comment_body}
File: {file_path}
Code context:
{code_around_comment}
```

## Monitor Cycle Flow

```
Every 5 minutes:
  │
  ├── git pull
  │   ├── SUCCESS ──▶ continue
  │   └── MERGE CONFLICT ──▶ LLM resolves conflicts ──▶ commit + push
  │
  ├── Detect project type
  │
  ├── Run build
  │   ├── SUCCESS ──▶ continue
  │   └── FAILURE ──▶ Compile Fixer Agent (up to 10 retries) ──▶ commit + push
  │                   └── 10 failures ──▶ give up, log reason
  │
  ├── Run tests
  │   ├── SUCCESS ──▶ continue
  │   ├── FAILURE ──▶ Test Agent fix (up to 10 retries) ──▶ commit + push
  │   │               └── 10 failures ──▶ give up, log reason
  │   └── GAPS ──▶ LLM reads all code + PR diff, writes missing tests ──▶ commit + push
  │
  ├── Check PR comments
  │   ├── No new comments ──▶ continue
  │   └── New comments ──▶ Comment Agent ──▶ gh pr comment (reply)
  │
  └── Print status summary
```

## File Structure

```
agent-pr-monitor/
├── Cargo.toml
├── build.sh
├── design-doc.md
├── src/
│   ├── main.rs
│   ├── pr.rs
│   ├── monitor.rs
│   ├── agents.rs
│   ├── compiler.rs
│   ├── tester.rs
│   ├── commenter.rs
│   ├── detect.rs
│   ├── state.rs
│   └── server.rs
└── frontend/
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── src/
    │   ├── main.tsx
    │   ├── App.tsx
    │   ├── api/
    │   │   └── client.ts
    │   ├── hooks/
    │   │   └── useSSE.ts
    │   ├── components/
    │   │   ├── Header.tsx
    │   │   ├── CounterCards.tsx
    │   │   ├── ActivityLog.tsx
    │   │   ├── FileExplorer.tsx
    │   │   ├── FileViewer.tsx
    │   │   ├── CommentThreads.tsx
    │   │   ├── CommentThread.tsx
    │   │   └── AgentLogs.tsx
    │   └── types/
    │       └── index.ts
    └── dist/
```

## Cargo.toml Dependencies

- `serde` + `serde_json` - JSON parsing for gh API responses and REST API
- `tokio` - async runtime (required by actix-web)
- `actix-web` - HTTP server for the dashboard
- `actix-cors` - CORS middleware (dev convenience)
- `rust-embed` - embed frontend/dist/ into the binary
- `mime_guess` - content-type detection for embedded static files
- No LLM crates - all LLM calls go through `std::process::Command` / `tokio::process::Command`

## Error Handling

- If `gh` CLI is not installed or not authenticated, print error and exit
- If LLM CLI is not installed, print error and exit
- If git pull fails (merge conflict), LLM reads the conflicted files and resolves them, commits and pushes
- If LLM returns garbage, retry once with a more explicit prompt
- If any fix (compile or test) fails 10 times, give up and log the reason why
- If push fails (permissions), print error and continue monitoring

## Build

Single command to build everything:
```
$ ./build.sh
```

`build.sh` runs: `cd frontend && bun install && bun run build && cd .. && cargo build --release`

## Limitations

- Requires `gh` CLI installed and authenticated
- Requires at least one LLM CLI installed (claude, gemini, copilot, or codex)
- Only monitors one PR at a time per process
- Clone goes to /tmp so it is ephemeral
- 30-second timeout per LLM call
