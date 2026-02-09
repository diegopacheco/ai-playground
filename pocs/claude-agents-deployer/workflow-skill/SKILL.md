---
name: deployer-workflow
description: Full-stack development workflow that orchestrates all deployer agents (backend, frontend, database, testing, review, documentation) in a phased pipeline. Asks which backend language (Java/Go/Rust) then runs all agents.
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Task, AskUserQuestion, TaskCreate, TaskUpdate, TaskList]
---

# Deployer Workflow Skill

Orchestrate all deployer agents in a structured, phased development pipeline.

## Instructions

When this skill is invoked, follow this exact workflow:

### Step 0 - Read the user prompt

The user will provide a prompt describing what they want to build: $ARGUMENTS

If no arguments are provided, use AskUserQuestion to ask: "What do you want to build? Describe the feature or application."

### Step 1 - Pick Backend Language

Use AskUserQuestion to ask which backend language to use. The options are:
- **Java** - Spring Boot 4.x, Java 25
- **Go** - Go 1.25+, Gin Gonic
- **Rust** - Rust 1.93+, Axum/Actix-web

### Step 2 - Read Agent Definitions

Read all agent definition files from the project's `agents/` directory. These files contain the instructions for each agent role. The agents directory is at the project root.

### Step 3 - Phase 1: Build (parallel)

**Step 3.1 - Write Design Document**

Before implementing, create a `design-doc.md` file in the project root that describes:
- High-level architecture overview
- Backend API endpoints and their responsibilities
- Frontend components and their interactions
- Database schema design (tables, relationships)
- Integration points between frontend, backend, and database

This design document will guide all the build agents and ensure consistency.

**Step 3.1.1 - Review Workflow Plan**

Present the user with all phases and steps that will be executed. Use AskUserQuestion to show the following checkboxes (all marked by default) and allow the user to uncheck any phases they want to skip:

```
The following phases will be executed. Uncheck any you want to skip:

- [x] Phase 1: Build
  - [x] Step 3.2 - Build Components (Backend, Frontend, Database)
  - [x] Step 3.3 - Verify Components
- [x] Phase 2: Test
  - [x] Unit Tests
  - [x] Integration Tests
  - [x] UI Tests (Playwright)
  - [x] Stress Tests (K6)
- [x] Phase 3: Review
  - [x] Code Review
  - [x] Security Review
- [x] Phase 4: Document
  - [x] Design Doc Sync
  - [x] Feature Documentation
  - [x] Changes Summary
- [x] Changelog & README Update
```

Store the user's selections and skip any unchecked phases/steps during execution.

**Step 3.1.2 - Initialize Progress Tracking**

Create a `todo.md` file in the project root with the current date in `yyyy-MM-dd` format and all selected phases listed with `[ ]` checkboxes. This file will be updated at the end of each phase to track progress.

Example:
```
# Progress - 2026-02-09

## Phase 1: Build
- [ ] Build Components (Backend, Frontend, Database)
- [ ] Verify Components

## Phase 2: Test
- [ ] Unit Tests
- [ ] Integration Tests
- [ ] UI Tests (Playwright)
- [ ] Stress Tests (K6)

## Phase 3: Review
- [ ] Code Review
- [ ] Security Review

## Phase 4: Document
- [ ] Design Doc Sync
- [ ] Feature Documentation
- [ ] Changes Summary

## Finalize
- [ ] Changelog
- [ ] README Update
```

**Step 3.2 - Build Components (parallel)**

Spawn 3 Task subagents in parallel using `subagent_type: "general-purpose"`. Each subagent receives the user's feature description, the design-doc.md content, plus the corresponding agent definition as context:

1. **Backend Developer** - Use the chosen language agent (java-backend-developer-agent.md, go-backend-developer-agent.md, or rust-backend-developer-agent.md). Tell it to implement the backend for the user's request following the design-doc.md.
2. **React Developer** - Use react-developer-agent.md. Tell it to implement the frontend for the user's request following the design-doc.md.
3. **Relational DBA** - Use relational-dba-agent.md. Tell it to design and create the database schema for the user's request following the design-doc.md.

**Step 3.3 - Verify Components**

After all 3 build subagents complete, verify that each component is working:

1. **Verify Database** - Check that the database schema was created and migrations run successfully. Start the database container if needed.
2. **Verify Backend** - Build and start the backend server. Verify it compiles/runs without errors and can connect to the database. Test a health endpoint if available.
3. **Verify Frontend** - Build the React frontend. Verify it compiles without errors. Check that it can be served and connects to the backend API.

Fix any issues before moving to Phase 2. All three components must be operational.

**Update Progress**: Update `todo.md` marking Phase 1 items as complete with `[x]`.

### Step 4 - Phase 2: Test (parallel)

Spawn 4 Task subagents in parallel:

1. **Unit Testers** - Use unit-testers-agent.md. Tell it to write unit tests for all the code written in Phase 1.
2. **Integration Tester** - Use integration-tester-agent.md. Tell it to write integration tests covering the API and database interactions.
3. **UI Testing Playwright** - Use ui-testing-playwright-agent.md. Tell it to write Playwright end-to-end tests for the React frontend.
4. **K6 Stress Test** - Use k6-stress-test-agent.md. Tell it to write k6 performance tests for the API endpoints.

Wait for all 4 to complete before moving to Phase 3.

**Update Progress**: Update `todo.md` marking Phase 2 items as complete with `[x]`.

### Step 5 - Phase 3: Review (parallel)

Create a `review/` folder with a subfolder named as the current date in `yyyy-MM-dd` format (e.g., `review/2026-02-09/`).

Spawn 2 Task subagents in parallel:

1. **Code Reviewer** - Use code-reviewer-agent.md. Tell it to review all code written in Phases 1 and 2 for quality, bugs, and best practices. Output the review to `review/{current-date}/code-review.md`.
2. **Security Reviewer** - Use security-reviewer-agent.md. Tell it to review all code for security vulnerabilities, OWASP Top 10, and security best practices. Output the review to `review/{current-date}/sec-review.md`.

Wait for both to complete. If either reviewer finds critical issues, fix them before moving to Phase 4.

**Update Progress**: Update `todo.md` marking Phase 3 items as complete with `[x]`.

### Step 6 - Phase 4: Document (parallel)

Use the same `review/{current-date}/` folder created in Phase 3.

Spawn 3 Task subagents in parallel:

1. **Design Doc Syncer** - Use design-doc-syncer-agent.md. Tell it to sync the design document with the implemented code, updating it to reflect what was built.
2. **Feature Documenter** - Use feature-documenter-agent.md. Tell it to document the feature including API docs, configuration, and usage. Output to `review/{current-date}/features.md`.
3. **Changes Summarizer** - Use changes-sumarizer-agent.md. Tell it to summarize all changes made, categorize them. Output to `review/{current-date}/summary.md`.

**Update Progress**: Update `todo.md` marking Phase 4 items as complete with `[x]`.

### Step 7 - Changelog

After all phases complete, create a `changelog.md` file in the project root.

Use git to analyze all changes made during this workflow:
- Run `git status` to see modified/added files
- Run `git diff` to see the actual code changes
- Run `git log` to see commit history if any commits were made

The changelog.md should contain:
- Current date in `yyyy-MM-dd` format at the top
- What was built (backend, frontend, database)
- List of all files created/modified with brief descriptions
- Test coverage summary (unit, integration, UI, stress)
- Review findings and fixes applied
- Documentation generated
- Any remaining issues or recommendations

**Update Progress**: Update `todo.md` marking Changelog as complete with `[x]`.

### Step 8 - Update README

Update or create the `README.md` file in the project root with:

1. **Project Overview** - Brief description of what was built
2. **Links to Documentation**:
   - Link to `design-doc.md`
   - Link to `review/{current-date}/code-review.md`
   - Link to `review/{current-date}/sec-review.md`
   - Link to `review/{current-date}/features.md`
   - Link to `review/{current-date}/summary.md`
   - Link to `changelog.md`
3. **Summary Highlights** - Extract the best parts from `review/{current-date}/summary.md` including:
   - Key features implemented
   - Architecture highlights
   - Notable technical decisions
4. **Quick Start** - How to run the project (backend, frontend, database)

**Update Progress**: Update `todo.md` marking README Update as complete with `[x]`. All items should now be marked as `[x]`.

## Agent Definition Location

The agent definitions are markdown files in the `agents/` directory at the project root. Each file contains the role description, capabilities, and guidelines for that agent. Always read the actual file content and pass it as context to the Task subagent.

## Task Subagent Pattern

When spawning each Task subagent, use this pattern:
- `subagent_type: "general-purpose"`
- Include the full agent definition markdown in the prompt
- Include the user's feature description
- Include relevant context from previous phases (file paths created, schemas designed, etc.)
- Tell the subagent to write actual code, not just describe what to do

## Phase Dependencies

- Phase 2 depends on Phase 1 (tests need code to test)
- Phase 3 depends on Phase 1 and Phase 2 (reviews need all code)
- Phase 4 depends on all previous phases (docs need final code)

Within each phase, all subagents run in parallel since they are independent of each other.
