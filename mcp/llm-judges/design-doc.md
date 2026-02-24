# LLM Judges MCP - Design Document

## Overview

LLM Judges is an MCP server written in Rust that sends content to 4 LLM judges (Claude, Codex, Copilot, Gemini) via their CLI tools, collects their verdicts, and returns a consolidated judgment. When a user asks to "fact check", "judge", or "validate" content from their current session, the MCP fans out the request to all 4 judges in parallel, each judge evaluates independently, and the MCP aggregates results into a final verdict.

## Technology Stack

- Rust 2024 edition (1.93)
- tokio (async runtime + subprocess management)
- serde / serde_json (serialization)
- MCP protocol over stdio (JSON-RPC 2.0)

No web framework needed. No database. No frontend. Pure stdio MCP server.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Claude Code / Codex CLI                  │
│         (calls MCP tools via stdio JSON-RPC)          │
└──────────────────────┬──────────────────────────────┘
                       │ stdin/stdout (JSON-RPC 2.0)
                       ▼
┌─────────────────────────────────────────────────────┐
│                  LLM Judges MCP Server                │
├─────────────────────────────────────────────────────┤
│  Router  │  Judge Engine  │  Aggregator               │
└──────────┴───────┬───────┴────────────────────────┘
                   │ parallel CLI subprocesses
                   ▼
┌─────────────────────────────────────────────────────┐
│                    Judge Runners                      │
├─────────────────────────────────────────────────────┤
│  Claude  │  Codex  │  Copilot  │  Gemini              │
│  (cli)   │  (cli)  │  (cli)    │  (cli)               │
└──────────┴─────────┴───────────┴──────────────────┘
```

## How It Works

1. User tells Claude Code: "fact check this" or "judge this code" or "validate this approach"
2. Claude Code calls the MCP tool `judge` with the content to evaluate
3. The MCP server builds a judge prompt wrapping the content
4. Spawns 4 parallel CLI subprocesses (claude, codex, copilot, gemini)
5. Each judge returns a structured verdict: PASS/FAIL/UNCERTAIN + reasoning
6. The aggregator collects all 4 verdicts and produces a final result
7. Returns the consolidated judgment to Claude Code

## MCP Tools

### Tool: `judge`

Sends content to all 4 LLM judges for evaluation.

**Input Schema**:
```json
{
  "content": "string - the content to be judged",
  "criteria": "string (optional) - specific criteria to judge against, defaults to general fact-checking"
}
```

**Output**:
```json
{
  "verdict": "PASS | FAIL | SPLIT",
  "score": "3/4",
  "judges": [
    {
      "name": "claude",
      "verdict": "PASS",
      "confidence": "high",
      "reasoning": "The claims are accurate because..."
    },
    {
      "name": "codex",
      "verdict": "PASS",
      "confidence": "medium",
      "reasoning": "Generally correct, though..."
    },
    {
      "name": "copilot",
      "verdict": "FAIL",
      "confidence": "low",
      "reasoning": "The third claim is incorrect..."
    },
    {
      "name": "gemini",
      "verdict": "PASS",
      "confidence": "high",
      "reasoning": "All statements check out..."
    }
  ],
  "summary": "3 out of 4 judges ruled PASS. Majority verdict: PASS."
}
```

### Tool: `judge_pick`

Same as `judge` but lets the user pick which judges to use.

**Input Schema**:
```json
{
  "content": "string - the content to be judged",
  "criteria": "string (optional) - specific criteria",
  "judges": ["claude", "gemini"]
}
```

**Output**: Same structure as `judge`, but only with selected judges.

### Tool: `list_judges`

Returns available judges and their status.

**Input Schema**: (none)

**Output**:
```json
{
  "judges": [
    { "name": "claude", "cli": "claude", "available": true },
    { "name": "codex", "cli": "codex", "available": true },
    { "name": "copilot", "cli": "copilot", "available": true },
    { "name": "gemini", "cli": "gemini", "available": true }
  ]
}
```

## Judge Prompt Template

```
You are an impartial judge evaluating the following content.

CRITERIA: {criteria or "Check for factual accuracy, logical consistency, and correctness."}

CONTENT TO JUDGE:
---
{content}
---

Evaluate the content and respond in this exact format:
VERDICT: PASS or FAIL or UNCERTAIN
CONFIDENCE: high or medium or low
REASONING: Your explanation in 2-3 sentences.
```

## Verdict Aggregation Rules

- 4 PASS = PASS
- 3 PASS + 1 anything = PASS
- 2 PASS + 2 FAIL = SPLIT
- 1 PASS + 3 FAIL = FAIL
- 0 PASS = FAIL
- UNCERTAIN counts are noted but treated as abstentions for majority calculation

## Project Structure

```
llm-judges/
├── Cargo.toml
├── design-doc.md
├── install.sh
├── README.md
├── src/
│   ├── main.rs              (stdio MCP server loop)
│   ├── mcp.rs               (JSON-RPC 2.0 protocol handling)
│   ├── tools.rs             (tool definitions and routing)
│   ├── engine.rs            (parallel judge execution + aggregation)
│   └── judges/
│       ├── mod.rs            (judge trait + registry)
│       ├── runner.rs         (CLI subprocess runner, reused pattern from agent-debate-club)
│       ├── claude.rs         (claude CLI command builder)
│       ├── codex.rs          (codex CLI command builder)
│       ├── copilot.rs        (copilot CLI command builder)
│       └── gemini.rs         (gemini CLI command builder)
```

## CLI Commands Per Judge

Reusing the exact same CLI patterns from agent-debate-club:

| Judge   | Command                                                          |
|---------|------------------------------------------------------------------|
| Claude  | `claude -p "{prompt}" --model opus --dangerously-skip-permissions` |
| Codex   | `codex exec --full-auto --model gpt-5.2 "{prompt}"`             |
| Copilot | `copilot --allow-all --model claude-sonnet-4 -p "{prompt}"`     |
| Gemini  | `gemini -y -p "{prompt}"`                                        |

Each subprocess gets a 120 second timeout. If a judge times out, its verdict is marked as TIMEOUT and excluded from aggregation.

## MCP Protocol (stdio JSON-RPC 2.0)

The server communicates over stdin/stdout using the MCP protocol:

**Initialize**: Server responds with capabilities (tools list).

**tools/list**: Returns the 3 tools (judge, judge_pick, list_judges) with their input schemas.

**tools/call**: Executes the requested tool and returns the result.

The server runs as a long-lived process, reading JSON-RPC messages from stdin and writing responses to stdout. Logs go to stderr.

## Install Script (install.sh)

Follows the same pattern as graph-postgres-mcp:

1. `cargo build --release` in the project directory
2. Verify the binary exists at `target/release/llm-judges`
3. Register in Claude Code: `claude mcp add llm-judges -s user -- {binary_path}`
4. Register in Codex CLI: append to `~/.codex/config.toml`

## Error Handling

- Judge CLI not found: mark judge as unavailable, skip it
- Judge timeout (120s): kill process, mark as TIMEOUT
- Judge returns unparseable output: mark as ERROR with raw output
- All judges fail: return error with details per judge
- Stdin/stdout broken: exit gracefully

## Dependencies (Cargo.toml)

```toml
[package]
name = "llm-judges"
version = "0.1.0"
edition = "2024"

[dependencies]
tokio = { version = "1", features = ["full", "process"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Minimal dependencies. No web framework. No database. Just tokio for async subprocess management and serde for JSON.
