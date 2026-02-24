# MCP Projects

This directory contains MCP-related projects and integrations built around Claude Code, Playwright, Flux, and Postgres.

## List of MCPs

### 1. `graph-postgres-mcp`

GraphQL-based MCP server for PostgreSQL.

What it does:
- Exposes a read-only MCP server for PostgreSQL data.
- Adds a GraphQL layer in front of Postgres so an AI client can query database data through GraphQL.
- Includes scripts to start/stop a local Postgres stack, install/uninstall the MCP, and validate the setup.

Main files:
- `graph-postgres-mcp/src/index.ts`
- `graph-postgres-mcp/src/schema-generator.ts`
- `graph-postgres-mcp/src/resolvers.ts`

### 2. `flux-kanban-mcp`

Flux MCP integration for Kanban-style task management in Claude Code.

What it does:
- Registers the Flux MCP server in Claude Code using a `podman` command (`mcp.json`).
- Starts a Flux web UI locally and connects it to Claude Code through MCP.
- Supports managing tasks/status in Flux while building software, with a recorded workflow in this project.

Main files:
- `flux-kanban-mcp/mcp.json`
- `flux-kanban-mcp/install-mcp-claude.sh`
- `flux-kanban-mcp/flux.sh`

### 3. `claude-code-mcp-playwright-fun`

Playwright MCP usage project for UI testing workflows in Claude Code.

What it does:
- Shows how to install and use the Playwright MCP in Claude Code.
- Uses MCP prompts to test a UI feature list and generate a test report.
- Contains generated Playwright tests and scripts for a product manager UI app.

Main files:
- `claude-code-mcp-playwright-fun/README.md`
- `claude-code-mcp-playwright-fun/tests/ui.spec.ts`
- `claude-code-mcp-playwright-fun/test-ui.sh`

### 4. `llm-judges`

LLM-as-judge MCP server design for multi-model validation.

What it does:
- Defines an MCP server that sends content to multiple LLM judge CLIs in parallel (Claude, Codex, Copilot, Gemini).
- Aggregates verdicts into a consolidated PASS/FAIL/SPLIT result.
- Specifies MCP tools for judging content, selecting judges, and listing available judges.

Current status:
- This folder currently contains the design document/specification for the MCP server implementation.

Main files:
- `llm-judges/design-doc.md`

## Notes

- `graph-postgres-mcp` is a custom MCP server implementation.
- `flux-kanban-mcp` is an MCP integration/setup around Flux.
- `claude-code-mcp-playwright-fun` is a Playwright MCP usage project and testing workflow.
- `llm-judges` is an LLM-as-judge MCP server design/specification.
