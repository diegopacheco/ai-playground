---
name: deployer-workflow
description: Full-stack development workflow that orchestrates deployer agents (backend, frontend, database, testing, review) in a phased pipeline. Asks which backend language (Java/Go/Rust) then runs all agents.
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Task, AskUserQuestion, TaskCreate, TaskUpdate, TaskList]
---

# Deployer Workflow Skill V4

On the very first message, display: `Deployer Workflow Skill V4 | Location: skills/workflow-skill/SKILL.md`

## Global Context
- User request: $ARGUMENTS
- Agents directory: `agents/`
- Agent definition files: all markdown files in `agents/`
- Design doc: `design-doc.md` (project root)
- Progress file: `todo.md` (project root)
- Mistakes file: `mistakes.md` (project root)
- Review folder: `review/{yyyy-MM-dd}/`
- Changelog: `changelog.md` (project root)
- README: `README.md` (project root)

## Agents

There are 5 agents total:
1. **Backend Developer** (java-backend-developer-agent.md / go-backend-developer-agent.md / rust-backend-developer-agent.md)
2. **React Developer** (react-developer-agent.md)
3. **Relational DBA** (relational-dba-agent.md)
4. **Testing Agent** (testing-agent.md) - handles unit tests, integration tests, UI tests (Playwright), stress tests (K6)
5. **Reviewer Agent** (reviewer-agent.md) - handles code review, security review, design doc sync, feature docs, changes summary

## Rules
- If $ARGUMENTS is empty, ask: "What do you want to build? Describe the feature or application."
- Ask backend language using AskUserQuestion with options:
  - Java: Spring Boot 4.x, Java 25
  - Go: Go 1.25+, Gin Gonic
  - Rust: Rust 1.93+, Axum/Actix-web
- Read all agent definitions from `agents/` and pass them as context to subagents.
- Use Task subagents with `subagent_type: "general-purpose"`.
- Include user request, relevant files, and the full agent definition in each subagent prompt.
- Phase order: Build -> Test -> Review (includes Changelog and README).
- Phase dependencies: Test depends on Build. Review depends on Build + Test.
- After each phase, update `todo.md` by marking completed items with `[x]`.

## Mistakes Tracking

- `mistakes.md` in the project root tracks all build failures, test failures, and issues across all phases.
- Every agent MUST read `mistakes.md` before starting work to avoid repeating past mistakes.
- Every agent MUST append new mistakes/issues they encounter to `mistakes.md` with the phase name and a short description.
- At the start of the workflow, create `mistakes.md` if it does not exist with a header `# Mistakes Log`.
- Format: `- [Phase N: Name] description of the mistake or issue and how it was fixed`

## Build and Test Enforcement

- The build MUST compile and pass before moving to Phase 2 (Test).
- ALL tests (unit, integration, UI, stress) MUST pass before moving to Phase 3 (Review).
- If the build fails, fix it immediately. Do not proceed until the build is green.
- If any test fails, debug and fix the root cause. Re-run all tests until they all pass.
- After fixing build or test failures, record what went wrong and how it was fixed in `mistakes.md`.
- At the end of Phase 1, run the full build and verify it succeeds.
- At the end of Phase 2, run all tests and verify they all pass.

## Step 1: Review Plan
Use AskUserQuestion with checkboxes (all checked by default) and allow unchecking. Store selections and skip unchecked items:
- Phase 1: Build
  - Build Components (Backend, Frontend, Database)
  - Verify Build (compile, run, connectivity)
- Phase 2: Test
  - All Tests (Unit, Integration, UI Playwright, K6 Stress)
- Phase 3: Review
  - Full Review (Code, Security, Design Doc Sync, Feature Docs, Changes Summary)
  - Changelog & README Update

Initialize `todo.md` with current date (yyyy-MM-dd) and all selected items as `[ ]`.

## Step 2: Design Doc
Create `design-doc.md` with:
- Architecture overview
- Backend API endpoints and responsibilities
- Frontend components and interactions
- Database schema design
- Integration points between frontend, backend, database

## Phase 1: Build
### Build Components (parallel)
Spawn 3 subagents in parallel:
1. **Backend Developer**: use chosen language agent (java/go/rust backend agent). Implement backend per `design-doc.md`. Agent MUST read `mistakes.md` first.
2. **React Developer**: use react-developer-agent.md. Implement frontend per `design-doc.md`. Agent MUST generate frontend unit tests (e.g. component tests using Vitest/Jest + React Testing Library). Agent MUST read `mistakes.md` first.
3. **Relational DBA**: use relational-dba-agent.md. Design and create DB schema per `design-doc.md`. Agent MUST read `mistakes.md` first.

### Verify Build
After all 3 subagents complete:
- Verify DB schema/migrations apply successfully.
- Verify backend compiles, builds, and runs with DB connection.
- Verify frontend builds and can be served and connects to backend.
- Run `build.sh` and check stdout/stderr for both backend and frontend. There MUST be zero errors AND zero warnings. If there are any errors or warnings, fix the root cause and re-run `build.sh` in a loop until the output is 100% clean (no errors, no warnings).
- If ANY build step fails: fix the issue, record it in `mistakes.md`, and re-verify.
- Do NOT proceed to Phase 2 until the full build is green and `build.sh` produces zero errors and zero warnings.

## Phase 2: Test
Spawn 1 subagent:
1. **Testing Agent**: use testing-agent.md. Run all tests (unit, integration, UI Playwright, K6 stress) for both backend AND frontend code from Phase 1. Frontend unit tests MUST be included. Agent MUST read `mistakes.md` first.

After the testing agent completes:
- Verify ALL tests pass by running the full test suite.
- If ANY test fails: fix the issue, record it in `mistakes.md`, re-run all tests.
- Do NOT proceed to Phase 3 until all tests are green.

### Verify Tests
- Verify that a dedicated script exists for each test type: `run-unit-tests.sh`, `run-integration-tests.sh`, `run-e2e-tests.sh`, `run-stress-tests.sh`. If any script is missing, create it.
- Run every test script and check stdout/stderr for both backend and frontend. There MUST be zero errors AND zero warnings. If there are any errors or warnings, fix the root cause and re-run the failing script in a loop until the output is 100% clean (no errors, no warnings).
- ALL test scripts MUST pass before proceeding. Do NOT move to Phase 3 until every test script exits with code 0 and produces zero errors and zero warnings.

## Phase 3: Review
Create `review/{current-date}/`.
Spawn 1 subagent:
1. **Reviewer Agent**: use reviewer-agent.md. Perform code review, security review, sync design doc, write feature docs, and summarize changes. Agent MUST read `mistakes.md` first. Outputs:
   - `review/{current-date}/code-review.md`
   - `review/{current-date}/sec-review.md`
   - `review/{current-date}/features.md`
   - `review/{current-date}/summary.md`
   - Updates `design-doc.md`

If critical issues are found, fix them and record in `mistakes.md`.

### Changelog
Create `changelog.md` using git info:
- Use `git status`, `git diff`, `git log`.
- Include current date, what was built, files created/modified, test coverage summary, review findings and fixes, docs generated, remaining issues or recommendations.

### README
Update/create `README.md` with:
- Overview
- Links to: `design-doc.md`, `review/{current-date}/code-review.md`, `review/{current-date}/sec-review.md`, `review/{current-date}/features.md`, `review/{current-date}/summary.md`, `changelog.md`
- Summary highlights from `review/{current-date}/summary.md`
- Quick start (backend, frontend, database)
