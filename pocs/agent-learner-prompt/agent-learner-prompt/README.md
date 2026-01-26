# Agent Learner Prompt

A self-learning CLI agent that iteratively improves its prompts based on execution results. Runs 5 learning cycles per task with detailed reporting.

## Features

- Learns from successes (memory.txt)
- Avoids known failures (anti-pattern.txt)
- Maintains prompt version history (prompt.md)
- Runs 5 learning cycles per task
- Shows learnings and anti-patterns per cycle
- Interactive REPL mode for continuous learning
- Runs generated code in solutions/

## Build

```bash
./build-all.sh
```

## Usage

Single task mode:
```bash
./run.sh "Create a hello world web server in Python"
./run.sh --model opus "Build a REST API"
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
agent> :memory         # Show learnings
agent> :anti           # Show anti-patterns
agent> :prompts        # Show prompt history
agent> :clear          # Clear screen
agent> :quit           # Exit REPL
agent> Create a web server   # Start 5-cycle learning session
```

## Stop

```bash
./stop.sh
```

## Test

```bash
./test.sh
```

## Learning Cycle Output

Each task runs through 5 learning cycles with reports:

```
************************************************************
LEARNING CYCLE 1/5
************************************************************

Executing agent for cycle 1...

============================================================
CYCLE 1 REPORT
============================================================
Status: SUCCESS

Learnings acquired this cycle:
  + Task completed successfully with current approach
  + Generated code executed without errors

Anti-patterns identified this cycle:
  (none)
============================================================
```

## Session Summary

After all cycles complete:

```
############################################################
LEARNING SESSION SUMMARY
############################################################
Total cycles: 5
Successes: 4
Failures: 1

All learnings accumulated:
  + Task completed successfully
  + File generation approach worked

All anti-patterns identified:
  - Avoid long-running operations

Prompt versions created: 1
############################################################
```

## Scripts

- `build-all.sh` - Build the project
- `run.sh` - Execute agent with task or enter REPL
- `stop.sh` - Stop running agents
- `test.sh` - Run tests

## Files

- `memory.txt` - Accumulated learnings from successful runs
- `anti-pattern.txt` - Patterns to avoid from failures
- `prompt.md` - Current and past prompt versions
- `solutions/` - Generated code output directory
- `design-doc.md` - Architecture and design decisions
- `todo.txt` - Project task tracking
