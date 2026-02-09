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

Spawn 3 Task subagents in parallel using `subagent_type: "general-purpose"`. Each subagent receives the user's feature description plus the corresponding agent definition as context:

1. **Backend Developer** - Use the chosen language agent (java-backend-developer-agent.md, go-backend-developer-agent.md, or rust-backend-developer-agent.md). Tell it to implement the backend for the user's request.
2. **React Developer** - Use react-developer-agent.md. Tell it to implement the frontend for the user's request.
3. **Relational DBA** - Use relational-dba-agent.md. Tell it to design and create the database schema for the user's request.

Wait for all 3 to complete before moving to Phase 2.

### Step 4 - Phase 2: Test (parallel)

Spawn 4 Task subagents in parallel:

1. **Unit Testers** - Use unit-testers-agent.md. Tell it to write unit tests for all the code written in Phase 1.
2. **Integration Tester** - Use integration-tester-agent.md. Tell it to write integration tests covering the API and database interactions.
3. **UI Testing Playwright** - Use ui-testing-playwright-agent.md. Tell it to write Playwright end-to-end tests for the React frontend.
4. **K6 Stress Test** - Use k6-stress-test-agent.md. Tell it to write k6 performance tests for the API endpoints.

Wait for all 4 to complete before moving to Phase 3.

### Step 5 - Phase 3: Review (parallel)

Spawn 2 Task subagents in parallel:

1. **Code Reviewer** - Use code-reviewer-agent.md. Tell it to review all code written in Phases 1 and 2 for quality, bugs, and best practices.
2. **Security Reviewer** - Use security-reviewer-agent.md. Tell it to review all code for security vulnerabilities, OWASP Top 10, and security best practices.

Wait for both to complete. If either reviewer finds critical issues, fix them before moving to Phase 4.

### Step 6 - Phase 4: Document (parallel)

Spawn 3 Task subagents in parallel:

1. **Design Doc Syncer** - Use design-doc-syncer-agent.md. Tell it to sync the design document with the implemented code, updating it to reflect what was built.
2. **Feature Documenter** - Use feature-documenter-agent.md. Tell it to document the feature including API docs, configuration, and usage.
3. **Changes Summarizer** - Use changes-sumarizer-agent.md. Tell it to summarize all changes made, categorize them, and generate a changelog entry.

### Step 7 - Summary

After all phases complete, provide a final summary to the user:
- What was built (backend, frontend, database)
- Test coverage (unit, integration, UI, stress)
- Review findings and fixes applied
- Documentation generated
- Any remaining issues or recommendations

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