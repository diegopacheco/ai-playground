# Repo MCP — Design Doc

## 1. Goal

A reusable MCP server, **Repo MCP**, that lets Claude Code answer arbitrary
questions about one or more local copies of GitHub repositories without ever
loading the whole codebase into the context window.

A user installs the tool once with `install.sh`. From then on they can register
any GitHub URL; the tool clones the repo to a known location, keeps it fresh
once per day, and exposes a small set of MCP tools (`list_files`, `read_file`,
`grep`, `tree`, `git_log`, …) that Claude uses to navigate the repo on demand.

## 2. Non-Goals

- No embeddings, no vector DB, no semantic search.
- No web UI, no daemon, no background workers.
- No write access to the repos (read-only).
- No private-repo auth handling beyond what the user's local `git` already does
  (SSH keys / `gh auth` / credential helper).
- No multi-user / server deployment — strictly a local CLI tool for one user.

## 3. User Stories

1. **Install once.** I run `./install.sh`; my Claude Code now has a
   `repo-mcp` server registered globally.
2. **Add a repo.** I run `repo-mcp add` (or invoke an MCP tool); it asks for
   the GitHub URL, clones it to `~/.mcp/repo-tool/repos/<name>`, and remembers
   it.
3. **Ask questions.** In Claude Code I ask "where is auth handled in
   `acme/api`?". Claude calls `grep`, `read_file`, etc. against the registered
   repo and answers — without me pasting any files.
4. **Stay current.** The next day I ask another question; the server notices
   the repo hasn't been pulled in >24h and runs `git pull --ff-only`
   transparently before answering.
5. **List / remove.** I can list registered repos and remove one I no longer
   care about.

## 4. High-Level Architecture

```
┌────────────────────────────────┐
│  Claude Code (MCP client)      │
└──────────────┬─────────────────┘
               │ stdio (MCP)
               ▼
┌────────────────────────────────┐
│  repo-mcp server  (Node/TS)    │
│   - tool router                │
│   - lazy git-pull guard        │
│   - path/binary filtering      │
└──────────────┬─────────────────┘
               │ child_process: git, rg
               ▼
┌────────────────────────────────┐
│  ~/.mcp/repo-tool/             │
│    bin/repo-mcp                │
│    registry.json               │
│    repos/<name>/               │
│      .repo-tool.meta.json      │
│      <repo files…>             │
└────────────────────────────────┘
```

### Stack
- **Language:** TypeScript 6.x on Node ≥ 20.
- **MCP SDK:** `@modelcontextprotocol/sdk`.
- **Transport:** stdio (default Claude Code MCP mode).
- **External binaries required on PATH:** `git`, `rg` (ripgrep).
- **No DB.** State is two JSON files (`registry.json` + per-repo `meta.json`).
- **Platforms:** macOS and Linux. No Windows.
- **Clone strategy:** plain `git clone <url>` — no `--depth`, no `--single-branch`.
  Whatever GitHub serves by default is what we keep.

## 5. Filesystem Layout

Everything the tool produces at runtime lives under `~/.mcp/repo-tool/`. The
POC project directory (`pocs/repo-mcp-poc/`) only holds the source and
`install.sh`; **no cloned repo, registry, or state file is ever written
inside the project directory.**

```
~/.mcp/repo-tool/
├── bin/
│   └── repo-mcp                # Node entrypoint (executable)
├── package.json
├── node_modules/
├── registry.json               # list of registered repos
└── repos/
    └── <repo-name>/
        ├── .repo-tool.meta.json
        └── <cloned repo>
```

`<repo-name>` is derived from the GitHub URL (`owner__repo`, to avoid
collisions between two `repo` names from different owners).

### `registry.json`
```json
{
  "version": 1,
  "repos": [
    {
      "name": "anthropic__claude-code-docs",
      "url": "https://github.com/anthropic/claude-code-docs.git",
      "path": "$HOME/.mcp/repo-tool/repos/anthropic__claude-code-docs",
      "added_at": "2026-05-13T10:00:00Z"
    }
  ]
}
```

### `<repo>/.repo-tool.meta.json`
```json
{
  "last_pull_at": "2026-05-13T10:00:00Z",
  "last_pull_status": "ok",
  "default_branch": "main"
}
```

## 6. Install Flow (`install.sh`)

`install.sh` is the single entry point a new user runs.

Steps, in order, with hard-fail on any error:

1. **Pre-flight checks**
   - `node --version` ≥ 20
   - `git --version` present
   - `rg --version` present (print install hint for `brew install ripgrep` if
     missing)
2. **Create dirs:** `mkdir -p ~/.mcp/repo-tool/{bin,repos}`
3. **Seed `registry.json`:** if `~/.mcp/repo-tool/registry.json` does not
   exist, write `{"version": 1, "repos": []}`. If it already exists, leave it
   untouched (re-running `install.sh` must never wipe registered repos).
4. **Copy sources:** copy `package.json` and `src/` into
   `~/.mcp/repo-tool/`; run `npm ci --omit=dev` there.
5. **Write entrypoint:** `~/.mcp/repo-tool/bin/repo-mcp` — a small shebang
   script (`#!/usr/bin/env node`) that requires the built server.
6. **Register with Claude Code (global):**
   Patch `~/.claude.json` (the user-scope MCP config) to add:
   ```json
   {
     "mcpServers": {
       "repo-mcp": {
         "command": "$HOME/.mcp/repo-tool/bin/repo-mcp",
         "args": []
       }
     }
   }
   ```
   Idempotent: if the entry exists, leave it. `install.sh` resolves `$HOME`
   at install time before writing the JSON, since Claude Code's MCP config
   does not expand shell variables.
7. **First-run wizard (interactive prompts):**
   - "Add a GitHub repo now? [y/N]"
   - If yes → ask "GitHub URL:", clone it, append to `registry.json`.
   - Loop until user says no.
8. Print summary: where things were installed, how to add more repos later,
   how to uninstall.

An accompanying `uninstall.sh` removes the directory and the
`claude.json` entry.

## 7. MCP Tool Surface

Kept deliberately small. Every tool takes a `repo` argument (the registered
name) so a single server instance serves many repos.

| Tool         | Input                                            | Output                                                   |
|--------------|--------------------------------------------------|----------------------------------------------------------|
| `list_repos` | —                                                | array of `{name, url, last_pull_at}`                     |
| `add_repo`   | `{url, branch?}`                                 | `{name, path}` (omit `branch` → use repo's default)      |
| `remove_repo`| `{name}`                                         | `{removed: true}`                                        |
| `tree`       | `{repo, path?, depth?}` (default depth 3)        | text tree of the directory                               |
| `list_files` | `{repo, glob?}` (default `**/*`)                 | array of relative paths                                  |
| `read_file`  | `{repo, path, start_line?, end_line?}`           | file content (full file by default), line-numbered       |
| `grep`       | `{repo, pattern, glob?, context?, max_results?}` | `[{path, line, match, context_before, context_after}]`   |
| `git_log`    | `{repo, path?, limit?}` (default 20)             | `[{sha, author, date, subject}]`                         |
| `repo_info`  | `{repo}`                                         | `{name, url, default_branch, last_pull_at, head}`        |

There are **no artificial truncation caps** on `read_file` content or `grep`
match counts. The tools return what was asked for. The job of keeping the
main Claude Code context window small lives in the *client* (§8), not in
arbitrary server-side limits.

### Why this surface?
- It mirrors how a human would explore a repo: tree → grep → open file. A
  sub-agent driving this loop is excellent at it.
- The tools are deliberately *raw*: no synthesis, no compression, no LLM
  call inside the server. Whatever intelligence is needed lives in the
  caller.

## 8. Context-Window Strategy

The core promise is **"answer any question without bloating the main Claude
Code context window."**

This is achieved by a **separation of concerns**:

- **The MCP server's job:** read whatever files are needed, fully and
  accurately, with no artificial caps.
- **The client's job:** ensure those reads happen in a *sub-agent's* context,
  not in the main thread.

### Client pattern: drive `repo-mcp` from a sub-agent

In Claude Code, the user (or a slash command) invokes the built-in `Agent`
tool with `subagent_type=Explore` (or any other agent) and a prompt of the
form:

> "Using the `repo-mcp` MCP tools against repo `<name>`, answer: `<question>`.
> Read whatever files you need, then return a concise answer with file:line
> citations."

The sub-agent's own context absorbs the file content. Only the sub-agent's
final answer (typically a few hundred tokens) returns to the main thread.
A 5,000-line file can be fully read by the sub-agent without ever entering
the main context window.

### What the server does to support this

1. **Return real content.** `read_file` returns the entire file by default,
   line-numbered. `grep` returns every match (with configurable surrounding
   context lines). No silent truncation.
2. **Path filtering.** Every tool respects `.gitignore` and skips binary files
   and any path matching a built-in blocklist (`node_modules`, `dist`,
   `*.lock`, `*.min.js`, `.git`, large generated files). This is about
   *relevance*, not size — the sub-agent shouldn't waste tokens on lockfiles.
2. **Line numbers always returned with file content.** Makes followups
   precise (`read_file path=… start_line=120 end_line=180`) when the sub-agent
   does want to scope further.
3. **No "dump everything" tool.** There is no `get_all_files_concatenated` or
   equivalent. The sub-agent must still be deliberate about *which* files to
   read; the server just doesn't lie about their contents.

### What we explicitly are NOT doing
- Not implementing a server-side `ask_repo(question)` tool that runs its own
  Claude API loop. The client already has a sub-agent primitive; duplicating
  it server-side would mean an extra API key, extra cost, and extra latency.
- Not silently truncating tool output. If a user really invokes `read_file`
  on a 50 MB binary blob, that's their problem — the blocklist already
  prevents the common cases.

## 9. Lazy Daily Update

No cron, no launchd. The freshness check lives in the tool router.

On every tool call that targets a specific repo:

1. Load `<repo>/.repo-tool.meta.json`.
2. If `now - last_pull_at < 24h` → proceed.
3. Else:
   - Acquire a per-repo lockfile (`<repo>/.repo-tool.lock`) with a short
     timeout. If another process holds it, skip pulling and proceed (best
     effort).
   - Run `git -C <repo> pull --ff-only` with a 30s timeout.
   - On success: update `last_pull_at = now`, `last_pull_status = "ok"`.
   - On failure (network, conflict, detached HEAD, etc.): record
     `last_pull_status = "<reason>"`, **but still proceed** with the stale
     copy. Surface a `stale: true` field in the tool response so Claude can
     mention it if relevant.

Rationale: the daily-pull requirement is "best effort, transparent." The
worst outcome is serving day-old data, which is acceptable. There is no
explicit `update_repo` tool — the server is read-only from the user's
perspective; the lazy guard is the only path that touches `git`.

## 10. Failure Modes & Edge Cases

| Case                                  | Behavior                                                    |
|---------------------------------------|-------------------------------------------------------------|
| Repo name collision on `add_repo`     | Reject with suggestion to remove the old one first.         |
| Private repo, no creds                | `git clone` fails; surface the git error verbatim.          |
| `git pull` rejected (non-ff)          | Log, mark stale, continue with current checkout.            |
| Binary file passed to `read_file`     | Return `{error: "binary file"}` rather than garbage bytes.  |
| Pattern that matches 10k lines        | Return all matches; sub-agent decides how to narrow.        |
| Repo deleted from disk out-of-band    | Mark in registry as `missing`; tools return clear error.    |
| User asks about an un-registered repo | Tool returns list of registered repos + hint to `add_repo`. |
| Two Claude sessions hit same repo     | Lockfile serializes pulls; reads are concurrent-safe.       |

## 11. MVP 1

Scope of the first deliverable — everything below ships together as MVP 1:

- **Skeleton:** TS 6.x project, MCP server boots over stdio, `list_repos`
  works, `install.sh` registers the server globally with Claude Code.
- **Core tools:** `tree`, `list_files`, `read_file` (full file, line-numbered,
  no caps), `grep` (with surrounding context lines).
- **Registry:** `add_repo`, `remove_repo`, persistence in `registry.json`
  (seeded by `install.sh`), first-run wizard in `install.sh`.
- **Lazy update:** freshness check on each tool call, per-repo lockfile,
  read-only — no explicit update tool exposed.
- **Git / info tools:** `git_log`, `repo_info`.
- **Hygiene:** blocklist + `.gitignore` filtering, `uninstall.sh`, README
  with a usage transcript that shows the sub-agent pattern in action.

## 12. Success Criteria

- One install command followed by one repo URL is enough to ask Claude
  questions about that codebase.
- A multi-question session about a 50k-file repo can be answered through the
  sub-agent pattern without the main Claude Code context absorbing any file
  contents directly.
- Adding a second repo doesn't require restarting Claude Code.
- After 24h of idle, the next tool call transparently pulls the latest
  default branch.
