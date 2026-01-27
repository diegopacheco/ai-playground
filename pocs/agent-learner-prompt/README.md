# Agent Learner Prompt

A self-learning CLI agent that iteratively improves its prompts based on execution results. Runs 3 learning cycles per task with code review.

## Features

- Learns from successes (memory.txt)
- Avoids known failures (anti-pattern.txt)
- Maintains prompt version history (prompt.md)
- Configurable learning cycles (default: 3)
- Code review: architecture, design, security, tests
- Filters generic learnings, keeps specific ones
- 10s timeout for solution runs (web servers OK)
- Interactive REPL mode

## Build

```bash
./build-all.sh
```

## Usage

Single task mode:
```bash
./run.sh "Create a hello world web server in Python"
./run.sh --cycles 5 "Build a REST API"
./run.sh --model opus --cycles 2 "Quick task"
```

Interactive REPL mode:
```bash
./run.sh --repl
./run.sh              # Also enters REPL if no task provided
```

View learnings:
```bash
./run.sh --show-memory
./run.sh --show-anti-patterns
./run.sh --list-prompts
./run.sh --help
```

## REPL Commands

```
agent> :help           # Show help
agent> :cycles 5       # Set cycles to 5
agent> :cycles         # Show current cycles
agent> :memory         # Show learnings
agent> :anti           # Show anti-patterns
agent> :prompts        # Show prompt history
agent> :clear          # Clear screen
agent> :quit           # Exit REPL
agent> Create a web server   # Start learning session
```

## Learning Cycle Phases

Each cycle has 3 phases:

```
Phase 1: Execute agent to generate code
Phase 2: Run solution with 10s timeout
Phase 3: Review code for:
  - Architecture issues
  - Design issues
  - Code quality issues
  - Security vulnerabilities
  - Missing tests
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
  + Design passed review - patterns used correctly

Anti-patterns identified this cycle:
  - Security issue: Hardcoded API key in config.js
  - Test issue: No unit tests for API endpoints

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

- `memory.txt` - Accumulated learnings (specific, not generic)
- `anti-pattern.txt` - Patterns to avoid from failures
- `prompt.md` - Current and past prompt versions
- `solutions/` - Generated code output directory
- `design-doc.md` - Architecture and design decisions
- `todo.txt` - Project task tracking
