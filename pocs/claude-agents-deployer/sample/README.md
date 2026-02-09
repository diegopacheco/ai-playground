# Sample

1. Instal all sub-agents: `./run.sh`
2. open claude in `target/test` ask it to build something.
3. them ask to use the sub-agents.

# How many sub-agents?

```
⏺ Based on the Task tool configuration, I have 7 built-in sub-agent types:

1. Bash - Command execution specialist
2. general-purpose - Research, code search, multi-step tasks
3. statusline-setup - Configure status line settings
4. Explore - Fast codebase exploration (find files, search code)
5. Plan - Software architect for implementation plans
6. claude-code-guide - Answers questions about Claude Code, Agent SDK, and Claude API

Plus there are 20 custom skill-based agents configured in your environment (visible in the skills list):

1. changes-sumarizer-agent
2. java-backend-developer-agent
3. k6-stress-test-agent
4. relational-dba-agent
5. security-reviewer-agent
6. react-developer-agent
7. unit-testers-agent
8. integration-tester-agent
9. rust-backend-developer-agent
10. code-reviewer-agent
11. feature-documenter-agent
12. design-doc-syncer-agent
13. ui-testing-playwright-agent
14. design-docer-syncer-agent
15. go-backend-developer-agent
16. And several book/utility skills (book-all, book-index, book-references, etc.)

So 7 built-in + 20 custom = 27 total agents/skills available.
```

## Agents in Action

```bash
⏺ 4 agents finished (ctrl+o to expand)
   ├─ Bash (Integration test API endpoints) · 1 tool use · 5.2k tokens
   │  ⎿  Done
   ├─ Task (Code review backend+frontend) · 3 tool uses · 15.7k tokens
   │  ⎿  Done
   ├─ Task (Security review full app) · 6 tool uses · 20.5k tokens
   │  ⎿  Done
   └─ Bash (Review shell scripts) · 3 tool uses · 5.1k tokens
      ⎿  Done

⏺ Here's the full testing summary from 4 sub-agents running in parallel:

  ---
  1. Integration Tests (Bash agent) -- 12/12 PASSED
  ┌───────────────────────────────────┬────────────┐
  │               Test                │   Result   │
  ├───────────────────────────────────┼────────────┤
  │ GET /persons (initial)            │ PASS       │
  ├───────────────────────────────────┼────────────┤
  │ POST /persons (create Alice)      │ PASS - 201 │
  ├───────────────────────────────────┼────────────┤
  │ POST /persons (create Bob)        │ PASS - 201 │
  ├───────────────────────────────────┼────────────┤
  │ GET /persons (list all)           │ PASS       │
  ├───────────────────────────────────┼────────────┤
  │ GET /persons/{id} (fetch one)     │ PASS       │
  ├───────────────────────────────────┼────────────┤
  │ PUT /persons/{id} (update)        │ PASS - 200 │
  ├───────────────────────────────────┼────────────┤
  │ GET /persons/{id} (verify update) │ PASS       │
  ├───────────────────────────────────┼────────────┤
  │ DELETE /persons/{id}              │ PASS - 200 │
  ├───────────────────────────────────┼────────────┤
  │ GET /persons (post-delete count)  │ PASS       │
  ├───────────────────────────────────┼────────────┤
  │ GET /persons/999 (not found)      │ PASS - 404 │
  ├───────────────────────────────────┼────────────┤
  │ DELETE /persons/999 (not found)   │ PASS - 404 │
  ├───────────────────────────────────┼────────────┤
  │ Response body validation          │ PASS       │
  └───────────────────────────────────┴────────────┘

 2. Code Review (general-purpose agent)

  High: .unwrap() on all DB operations + Mutex poisoning cascade -- one panic kills the server permanently until restart.
  Medium: No input validation, no frontend error handling on mutations, no response status checking.
  Low: No pagination, hardcoded URLs, no delete confirmation, inline styles.

  3. Security Review (general-purpose agent)
  ┌─────────────────────────────────┬──────────────────────────────────────────────┐
  │             Finding             │                   Severity                   │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ SQL Injection                   │ Safe -- parameterized queries used correctly │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ XSS                             │ Safe -- React escapes by default             │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ No authentication/authorization │ High                                         │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ Wildcard CORS + No CSRF         │ Medium                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ .unwrap() panics (DoS vector)   │ Medium                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ No input validation             │ Medium                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ No HTTPS                        │ Medium                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ No rate limiting                │ Low                                          │
  ├─────────────────────────────────┼──────────────────────────────────────────────┤
  │ PID file race condition         │ Low                                          │
  └─────────────────────────────────┴──────────────────────────────────────────────┘

4. Shell Scripts Review (Bash agent) -- 9/9 PASS

Both run.sh and stop.sh have correct permissions, proper PID management, and handle edge cases (missing PID file, dead
processes).

---
Bottom line: All CRUD operations work correctly. The app is functionally solid. The findings are typical for a local dev
prototype -- no SQL injection or XSS risks. The .unwrap() / Mutex poisoning pattern is the most notable concern if this were
production code.

⏺ React Developer Review -- App.js

  Component structure: Single App component is appropriate for this scope. No over-engineering with unnecessary component
  splitting.

  State management:
  - useState for persons, form, editingId -- correct and minimal.
  - form state uses a single object with spread updates ({...form, name: e.target.value}) -- clean pattern.
  - resetForm() properly clears both form and editingId.

  useEffect at line 17: useEffect(() => { fetchPersons(); }, []) -- empty dependency array means fetch once on mount. Correct.
  React strict mode in dev will double-invoke this, but that's harmless for a GET.

  Data fetching patterns:
  - App.js:10-15 -- fetchPersons uses .then(r => r.json()).then(setPersons). Passing setPersons directly as callback is clean.
  - App.js:14 -- .catch(() => {}) swallows errors silently. Should at minimum console.error or set an error state.
  - App.js:25-26 -- After PUT, calls resetForm() then fetchPersons() in .then(). Does not check response.ok, so a 400/500 response
  still triggers form reset.
  - App.js:28-29 -- Same issue on POST.
  - App.js:34-35 -- DELETE has no .catch() at all. Unhandled promise rejection on network failure.

  Form handling:
  - App.js:21 -- parseInt(form.age) without radix parameter. Works in practice since type="number" constrains input, but
  parseInt(form.age, 10) is safer.
  - App.js:24 -- if (editingId) -- works because SQLite AUTOINCREMENT starts at 1, so id is never 0. Fragile but acceptable here.
  - HTML required attributes on all inputs prevent empty submissions -- correct.

  Rendering:
  - App.js:94 -- key={p.id} is correct. Uses stable unique identifier.
  - App.js:77-81 -- Cancel button only renders when editing -- proper conditional rendering.
  - Inline styles throughout -- functional, not ideal for maintainability but acceptable for this scope.

  Missing patterns (not bugs, just notes):
  - No loading indicator during fetches
  - No empty state message when persons list is empty
  - No error display to user

  ---
  Now the remaining relevant agents:

⏺ Skill(/security-reviewer-agent)
  ⎿  Successfully loaded skill

⏺ Security Review -- Person CRUD Application

  SQL Injection -- SAFE
  All queries in main.rs use rusqlite parameterized syntax (?1, ?2, ?3). No string concatenation or format macros for SQL. This is
  secure.

  XSS -- SAFE
  React's JSX {} rendering auto-escapes all content in App.js:96-99. No use of dangerouslySetInnerHTML. Safe by default.

  CORS -- Medium risk (main.rs:136-139)
  allow_any_origin().allow_any_method().allow_any_header() -- any website can call all endpoints including DELETE and PUT.
  Acceptable for localhost dev, problematic if exposed to network.

  No Authentication/Authorization -- High for production, expected for prototype
  Zero auth on all 5 endpoints. Combined with wildcard CORS, any browser tab on any website can perform full CRUD operations.

  No CSRF Protection -- Medium
  State-changing endpoints (POST, PUT, DELETE) accept requests with no CSRF tokens. CORS allow_any_origin means CORS provides zero
  CSRF mitigation.

  Input Validation -- Medium
  - main.rs:8-13 -- No length limits on name/email, no format validation on email, age accepts negative values and extremes (i32
  range).
  - DB has NOT NULL constraints but empty strings pass that check.

  Denial of Service via Panic -- Medium (main.rs:33,77,100,119)
  .unwrap() on all DB operations. Single failure poisons the std::sync::Mutex, permanently crashing all subsequent requests until
  process restart.

  No HTTPS (main.rs:149)
  Plain HTTP on 127.0.0.1:8080. Data in cleartext. Binding to localhost limits exposure.

  No Rate Limiting
  Unlimited POST requests could fill disk via SQLite. Unlimited GET against Mutex serialization amplifies contention.

  Shell Scripts -- Low risk
  - run.sh -- No set -e, so failed cargo build still tries to run binary. PIDs of failed processes get saved.
  - stop.sh -- Properly checks if process exists before killing. Handles missing PID file. Clean.

  Database file (main.rs:129)
  persons.db created relative to working directory. Could end up in unexpected locations.
  ┌─────────────────────────┬────────────────────┐
  │         Finding         │      Severity      │
  ├─────────────────────────┼────────────────────┤
  │ SQL Injection           │ Safe               │
  ├─────────────────────────┼────────────────────┤
  │ XSS                     │ Safe               │
  ├─────────────────────────┼────────────────────┤
  │ No Auth                 │ High (if deployed) │
  ├─────────────────────────┼────────────────────┤
  │ Wildcard CORS + No CSRF │ Medium             │
  ├─────────────────────────┼────────────────────┤
  │ No Input Validation     │ Medium             │
  ├─────────────────────────┼────────────────────┤
  │ Panic/DoS via .unwrap() │ Medium             │
  ├─────────────────────────┼────────────────────┤
  │ No HTTPS                │ Medium             │
  ├─────────────────────────┼────────────────────┤
  │ No Rate Limiting        │ Low                │
  ├─────────────────────────┼────────────────────┤
  │ Shell script no set -e  │ Low                │
  ├─────────────────────────┼────────────────────┤
  │ Relative DB path        │ Low                │
  └─────────────────────────┴────────────────────┘
  ---
  Now let me invoke the remaining relevant custom agents.

⏺ Skill(/relational-dba-agent)
  ⎿  Successfully loaded skill

⏺ Relational DBA Review -- SQLite Schema & Queries

  Schema (main.rs:20-28):
  CREATE TABLE IF NOT EXISTS persons (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      email TEXT NOT NULL,
      age INTEGER NOT NULL
  )

  Schema findings:

  - No UNIQUE constraint on email -- Duplicate emails are allowed. If email should be a natural key or unique identifier, this is a
   data integrity gap.
  - No CHECK constraint on age -- age INTEGER NOT NULL accepts -999, 0, or 2147483647. A CHECK(age >= 0 AND age <= 150) would
  enforce valid range at the DB level.
  - No length constraints -- SQLite doesn't enforce VARCHAR(n) but a CHECK(length(name) <= 255) could prevent abuse.
  - No indexes beyond PK -- If the table grows, queries filtering by email or name would do full table scans. For this scale it's
  fine.
  - AUTOINCREMENT -- SQLite's AUTOINCREMENT prevents rowid reuse after deletion. This is correct for a PK that may be referenced
  externally. Slight overhead vs plain INTEGER PRIMARY KEY but negligible.
  - No created_at/updated_at columns -- No audit trail. Acceptable for a prototype.

  Query analysis:
  Query: SELECT id, name, email, age FROM persons
  Location: Line 34
  Assessment: Full table scan, no WHERE, no LIMIT/OFFSET. Returns all rows in memory. Will degrade with large datasets.
  ────────────────────────────────────────
  Query: SELECT ... WHERE id = ?1
  Location: Line 54
  Assessment: Uses PK index. Optimal.
  ────────────────────────────────────────
  Query: INSERT INTO (name, email, age) VALUES (?1, ?2, ?3)
  Location: Line 74
  Assessment: Parameterized. Correct. last_insert_rowid() on line 78 is safe because Mutex ensures no concurrent inserts between
    execute and rowid fetch.
  ────────────────────────────────────────
  Query: UPDATE ... WHERE id = ?4
  Location: Line 97
  Assessment: Uses PK index. Checks affected rows for 404. Correct.
  ────────────────────────────────────────
  Query: DELETE FROM persons WHERE id = ?1
  Location: Line 118
  Assessment: Uses PK index. Checks affected rows for 404. Correct.
  Concurrency & Connection management:
  - Single Connection behind std::sync::Mutex -- all DB access is serialized. This is the simplest correct approach for SQLite
  which has limited write concurrency (one writer at a time). For read-heavy workloads, SQLite WAL mode + multiple read connections
   would improve throughput.
  - No WAL mode set -- SQLite defaults to journal mode delete. Enabling PRAGMA journal_mode=WAL would allow concurrent readers
  while writing.
  - No PRAGMA foreign_keys = ON -- not needed here (no foreign keys), but good practice to enable.

  Missing:
  - No VACUUM or maintenance strategy
  - No backup mechanism for persons.db
  - No pagination on the list query (GET /persons)

  ---
  Now the remaining relevant agents:

⏺ ---
Playwright UI Test Results: 9/9 PASSED
┌─────┬───────────────────────────────────────────────────────────────────────┬────────┐
│  #  │                                 Test                                  │ Result │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 1   │ Page loads with heading, form, empty table                            │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 2   │ Create person (Alice) -- row appears, form clears                     │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 3   │ Create second person (Bob) -- table shows 2 rows                      │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 4   │ Click Edit -- form populates, button becomes "Update", Cancel appears │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 5   │ Submit update -- name/age changed, table reflects update              │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 6   │ Cancel edit -- form clears, returns to "Add" mode                     │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 7   │ Delete person (Bob) -- row removed from table                         │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 8   │ Delete last person -- table becomes empty                             │ PASS   │
├─────┼───────────────────────────────────────────────────────────────────────┼────────┤
│ 9   │ Visual verification -- screenshots captured                           │ PASS   │
└─────┴───────────────────────────────────────────────────────────────────────┴────────┘
---
Grand Summary -- All Custom Agents Invoked
┌──────────────────────────────┬────────┬──────────────────────────────────────────────────────────────────────┐
│            Agent             │ Status │                              Key Result                              │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ code-reviewer-agent          │ Done   │ .unwrap() cascade is top concern, otherwise clean code               │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ rust-backend-developer-agent │ Done   │ std::sync::Mutex blocks async runtime, parameterized queries correct │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ react-developer-agent        │ Done   │ State management correct, missing error handling on fetches          │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ security-reviewer-agent      │ Done   │ SQL injection safe, XSS safe, wildcard CORS + no auth flagged        │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ relational-dba-agent         │ Done   │ No UNIQUE on email, no CHECK on age, no pagination                   │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ integration-tester-agent     │ Done   │ 26/26 API tests passed                                               │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ unit-testers-agent           │ Done   │ No unit tests exist (identified gap)                                 │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ changes-sumarizer-agent      │ Done   │ 8 files created, full stack documented                               │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ feature-documenter-agent     │ Done   │ All CRUD features documented with endpoints                          │
├──────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────┤
│ ui-testing-playwright-agent  │ Done   │ 9/9 browser UI tests passed with screenshots                         │
└──────────────────────────────┴────────┴──────────────────────────────────────────────────────────────────────┘
Agents not invoked (irrelevant to this project): java-backend-developer-agent, go-backend-developer-agent, k6-stress-test-agent,
design-doc-syncer-agent, design-docer-syncer-agent, and the book-related skills.

```

## Workflow in Action

```

 ▐▛███▜▌   Claude Code v2.1.37
▝▜█████▛▘  Opus 4.6 · Claude Pro
  ▘▘ ▝▝    ~/git/diegopacheco/ai-playground/pocs/claude-agents-deployer/sample

❯ /ad:wf

⏺ Let me read the skill file first.

⏺ Read 1 file (ctrl+o to expand)

⏺ The user didn't provide a description of what to build. Let me ask.

⏺ User answered Claude's questions:
  ⎿  · What do you want to build? Describe the feature or application. → Blog Platform

⏺ The user wants to build a Blog Platform. Now let me ask which backend language to use.

⏺ User answered Claude's questions:
  ⎿  · Which backend language should we use? → Rust

⏺ The user chose Rust. Now let me read all agent definition files from the agents/ directory.

⏺ Searched for 1 pattern (ctrl+o to expand)

⏺ Now let me read the Phase 1 agent definitions - Rust backend, React developer, and Relational DBA.

⏺ Read 3 files (ctrl+o to expand)

⏺ Now let me create the task tracking and start Phase 1. Let me set up tasks first.

⏺ Phase 1: Build (parallel)

  Launching 3 agents in parallel - Rust Backend, React Frontend, and Database.

⏺ 2 Task agents launched (ctrl+o to expand)
   ├─ Rust backend blog platform
   │  ⎿  Running in the background (shift+↑ to manage)
   └─ React frontend blog platform
      ⎿  Running in the background (shift+↑ to manage)

⏺ Phase 1 agents are running in parallel. Let me read the Phase 2 agent definitions while waiting, so we're ready to launch them as
   soon as Phase 1 completes.

⏺ Read 4 files (ctrl+o to expand)

⏺ Good, I've read all Phase 2 agent definitions. Now let me also pre-read the Phase 3 and Phase 4 agent definitions while waiting.

⏺ Read 5 files (ctrl+o to expand)

⏺ All agent definitions are loaded. All 3 Phase 1 agents are still running in parallel:

  - Rust Backend - Building Axum + SQLx blog API
  - React Frontend - Building React 19 + TanStack + Tailwind blog UI
  - Database - Setting up PostgreSQL 17 schema + podman scripts

  Waiting for all 3 to complete before starting Phase 2 (Testing)...

✻ Cooked for 1m 33s · 2 agents running in the background

⏺ Agent "Database schema blog platform" completed

⏺ The DB agent completed but couldn't write files. Let me create them directly while the other agents continue.

⏺ Bash(mkdir -p /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/claude-agents-deployer/sample/db)
  ⎿  (No content)

⏺ Write(db/schema.sql)
  ⎿  Wrote 31 lines to db/schema.sql
       1 CREATE TABLE users (
       2     id SERIAL PRIMARY KEY,
       3     name VARCHAR(255) NOT NULL,
       4     email VARCHAR(255) UNIQUE NOT NULL,
       5     created_at TIMESTAMP DEFAULT NOW()
       6 );
       7
       8 CREATE TABLE posts (
       9     id SERIAL PRIMARY KEY,
      10     title VARCHAR(255) NOT NULL,
     … +21 lines (ctrl+o to expand)

⏺ Write(db/start-db.sh)
  ⎿  Wrote 15 lines to db/start-db.sh
       1 #!/bin/bash
       2 podman run -d \
       3   --name blogdb-postgres \
       4   -e POSTGRES_DB=blogdb \
       5   -e POSTGRES_USER=bloguser \
       6   -e POSTGRES_PASSWORD=blogpass \
       7   -p 5432:5432 \
       8   docker.io/library/postgres:17
       9
      10 echo "Waiting for PostgreSQL to be ready..."
     … +5 lines (ctrl+o to expand)

⏺ Write(db/stop-db.sh)
  ⎿  Wrote 4 lines to db/stop-db.sh
      1 #!/bin/bash
      2 podman stop blogdb-postgres
      3 podman rm blogdb-postgres

⏺ Write(db/create-schema.sh)
  ⎿  Wrote 3 lines to db/create-schema.sh
      1 #!/bin/bash
      2 podman exec -i blogdb-postgres psql -U bloguser -d blogdb < "$(dirname "$0")/schema.sql"

⏺ Write(db/run-sql-client.sh)
  ⎿  Wrote 3 lines to db/run-sql-client.sh
      1 #!/bin/bash
      2 podman exec -it blogdb-postgres psql -U bloguser -d blogdb

⏺ Bash(chmod +x /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/claude-agents-deployer/sample/db/*.sh)
  ⎿  (No content)

⏺ Database files created and made executable. Still waiting for Rust Backend and React Frontend agents to finish...

  5 tasks (0 done, 1 in progress, 4 open) · ctrl+t to hide tasks
  ◼ Phase 1: Build - Rust Backend, React Frontend, Database
  ◻ Phase 2: Test - Unit, Integration, UI, Stress › blocked by #1
  ◻ Phase 3: Review - Code and Security › blocked by #1, #2
  ◻ Phase 4: Document - Design Doc, Feature Doc, Changes › blocked by #1, #2, #3
  ◻ Final Summary › blocked by #1, #2, #3, #4

```