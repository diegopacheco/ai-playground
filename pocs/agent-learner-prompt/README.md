# Agent Learner Prompt

A self-learning CLI agent that iteratively improves its prompts based on execution results. Runs 3 learning cycles per task with code review. Supports multiple agents: claude, codex, copilot, gemini.

## Features

- Multi-agent support (claude, codex, copilot, gemini)
- Easy agent/model switching via flags
- Per-project learning files (memory.txt, mistakes.txt, prompts.md)
- Maintains prompt version history (updated every cycle)
- Code review: architecture, design, security, tests
- Filters generic learnings, keeps specific ones
- 10s timeout for solution runs (web servers OK)
- Final code copied to code/ folder
- Interactive REPL mode

## Build

```bash
./build-all.sh
```

## Usage

Single task mode:
```bash
./run.sh "Create a hello world web server in Python"
./run.sh --agent claude --model opus "Build a REST API"
./run.sh --agent codex --model gpt-5.2 "Create CLI tool"
./run.sh --agent copilot "Use copilot agent"
./run.sh --agent gemini "Use gemini agent"
./run.sh --cycles 5 "Build a REST API"
```

Interactive REPL mode:
```bash
./run.sh --repl
./run.sh              # Also enters REPL if no task provided
```

## Supported Agents

| Agent | Default Model | CLI Command |
|-------|---------------|-------------|
| claude | sonnet | `claude -p <prompt> --model <model>` |
| codex | gpt-5.2 | `codex exec --full-auto --model <model>` |
| copilot | claude-sonnet-4 | `copilot --allow-all --model <model> -p` |
| gemini | gemini-2.5-pro | `gemini -y <prompt>` |

## REPL Commands

```
agent> :help           # Show help
agent> :agent claude   # Switch to claude agent
agent> :agent codex    # Switch to codex agent
agent> :model opus     # Switch model
agent> :cycles 5       # Set cycles to 5
agent> :memory         # Show learnings
agent> :mistakes       # Show mistakes to avoid
agent> :prompts        # Show prompt history
agent> :clear          # Clear screen
agent> :quit           # Exit REPL
agent> Create a web server   # Start learning session
```

## Learning Cycle Phases

Each cycle has 6 phases:

```
Phase 1: Execute agent to generate code
Phase 2: Run solution with 10s timeout
Phase 3: Review code for architecture, design, security, tests
Phase 4: Extract learnings from cycle (LLM)
Phase 5: Extract mistakes to avoid (LLM)
Phase 6: Improve prompt for next cycle (LLM)
```

## Project Structure

Each task creates a project folder:
```
solutions/{project}/
├── memory.txt       # Learnings accumulated
├── mistakes.txt     # Mistakes to avoid
├── prompts.md       # Prompt versions (updated every cycle)
├── cycle-1/         # First cycle output
├── cycle-2/         # Second cycle output
├── cycle-3/         # Third cycle output
└── code/            # Final production code
```

## Cycle Report

```
============================================================
CYCLE 1 REPORT
============================================================
Status: SUCCESS

Review findings:
  Architecture: OK
  Design: OK
  Code Quality: OK
  Security: Issues found
  Tests: Issues found

Learnings acquired this cycle:
  + Architecture passed review - structure is appropriate

Mistakes identified this cycle:
  - Security issue: Hardcoded API key in config.js

Prompt was improved and archived for next cycle
============================================================
```

## Stop

```bash
./stop.sh
```

## Test

```bash
./test.sh
```

## Scripts

- `build-all.sh` - Build the project
- `run.sh` - Execute agent with task or enter REPL
- `stop.sh` - Stop running agents
- `test.sh` - Run tests

## Files

- `solutions/` - Generated projects with per-project learning files
- `design-doc.md` - Architecture and design decisions
- `todo.txt` - Project task tracking
