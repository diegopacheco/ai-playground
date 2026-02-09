---
name: deployer-workflow
description: Full-stack development workflow that orchestrates all deployer agents (backend, frontend, database, testing, review, documentation) in a phased pipeline. Asks which backend language (Java/Go/Rust) then runs all agents.
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Task, AskUserQuestion, TaskCreate, TaskUpdate, TaskList]
---

# Deployer Workflow Skill

## Global Context
- User request: $ARGUMENTS
- Agents directory: `agents/`
- Agent definition files: all markdown files in `agents/`
- Design doc: `design-doc.md` (project root)
- Progress file: `todo.md` (project root)
- Review folder: `review/{yyyy-MM-dd}/`
- Changelog: `changelog.md` (project root)
- README: `README.md` (project root)

## Rules
- If $ARGUMENTS is empty, ask: "What do you want to build? Describe the feature or application."
- Ask backend language using AskUserQuestion with options:
  - Java: Spring Boot 4.x, Java 25
  - Go: Go 1.25+, Gin Gonic
  - Rust: Rust 1.93+, Axum/Actix-web
- Read all agent definitions from `agents/` and pass them as context to subagents.
- Use Task subagents with `subagent_type: "general-purpose"`.
- Include user request, relevant files, and the full agent definition in each subagent prompt.
- Phase order: Build -> Test -> Review -> Document -> Changelog -> README.
- Phase dependencies: Test depends on Build. Review depends on Build + Test. Document depends on all previous phases.
- After each phase, update `todo.md` by marking completed items with `[x]`.

## Step 1: Review Plan
Use AskUserQuestion with checkboxes (all checked by default) and allow unchecking. Store selections and skip unchecked items:
- Phase 1: Build
  - Build Components (Backend, Frontend, Database)
  - Verify Components
- Phase 2: Test
  - Unit Tests
  - Integration Tests
  - UI Tests (Playwright)
  - Stress Tests (K6)
- Phase 3: Review
  - Code Review
  - Security Review
- Phase 4: Document
  - Design Doc Sync
  - Feature Documentation
  - Changes Summary
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
1. Backend Developer: use chosen language agent (java/go/rust backend agent). Implement backend per `design-doc.md`.
2. React Developer: use react-developer-agent.md. Implement frontend per `design-doc.md`.
3. Relational DBA: use relational-dba-agent.md. Design and create DB schema per `design-doc.md`.

### Verify Components
Verify DB schema/migrations, backend builds and runs with DB connection, frontend builds and can be served and connects to backend. Fix issues before proceeding.

## Phase 2: Test (parallel)
Spawn 4 subagents:
1. Unit Testers: unit-testers-agent.md for all code from Phase 1.
2. Integration Tester: integration-tester-agent.md for API and DB interactions.
3. UI Testing Playwright: ui-testing-playwright-agent.md for React e2e tests.
4. K6 Stress Test: k6-stress-test-agent.md for API perf tests.

## Phase 3: Review (parallel)
Create `review/{current-date}/`.
Spawn 2 subagents:
1. Code Reviewer: code-reviewer-agent.md -> `review/{current-date}/code-review.md`.
2. Security Reviewer: security-reviewer-agent.md -> `review/{current-date}/sec-review.md`.
If critical issues are found, fix before Phase 4.

## Phase 4: Document (parallel)
Use `review/{current-date}/`.
Spawn 3 subagents:
1. Design Doc Syncer: design-doc-syncer-agent.md, update `design-doc.md`.
2. Feature Documenter: feature-documenter-agent.md -> `review/{current-date}/features.md`.
3. Changes Summarizer: changes-sumarizer-agent.md -> `review/{current-date}/summary.md`.

## Changelog
Create `changelog.md` using git info:
- Use `git status`, `git diff`, `git log`.
- Include current date, what was built, files created/modified, test coverage summary, review findings and fixes, docs generated, remaining issues or recommendations.

## README
Update/create `README.md` with:
- Overview
- Links to: `design-doc.md`, `review/{current-date}/code-review.md`, `review/{current-date}/sec-review.md`, `review/{current-date}/features.md`, `review/{current-date}/summary.md`, `changelog.md`
- Summary highlights from `review/{current-date}/summary.md`
- Quick start (backend, frontend, database)
