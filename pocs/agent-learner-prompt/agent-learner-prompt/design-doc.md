# Agent Learner Prompt - Design Document

## Overview

A self-learning CLI agent that iteratively improves its prompts based on execution results. The agent learns from successes and failures, storing knowledge in persistent files. Runs 5 learning cycles per task and supports interactive REPL mode.

## Architecture

```
agent-learner-prompt/
├── src/
│   └── main.rs          # CLI entry point and orchestration
├── memory.txt           # Accumulated learnings
├── anti-pattern.txt     # Bad practices to avoid
├── prompt.md            # Prompt version history
├── solutions/           # Generated code output
│   └── {project}/
│       ├── cycle-1/     # First cycle output
│       ├── cycle-2/     # Second cycle output
│       ├── cycle-3/     # Third cycle output
│       ├── cycle-4/     # Fourth cycle output
│       └── cycle-5/     # Fifth cycle output
├── build-all.sh         # Build the project
├── run.sh               # Execute the agent
├── stop.sh              # Stop running agents
└── test.sh              # Validate functionality
```

## Core Components

### 1. Prompt Manager
- Reads current prompt from prompt.md
- Archives prompts to Past Prompts section before updates
- Generates improved prompts based on learnings
- Timestamps each prompt version

### 2. Memory System
- memory.txt: Stores successful patterns and learnings
- anti-pattern.txt: Stores failures and patterns to avoid
- Both files are injected into the system prompt
- Updated after each learning cycle

### 3. Agent Executor
- Spawns claude CLI as subprocess
- Captures stdout/stderr
- Implements 5 learning cycles per task
- Timeout of 300 seconds per cycle

### 4. Learning Extractor
- Analyzes agent output for success/failure patterns
- Extracts actionable learnings per cycle
- Updates memory.txt or anti-pattern.txt accordingly
- Reports learnings and anti-patterns after each cycle

### 5. Solution Runner
- Executes generated code in solutions/{project}/cycle-N/
- Validates output by running run.sh if present
- Reports success/failure back to learning system

### 6. REPL Mode
- Interactive loop for continuous learning
- Commands: :quit, :memory, :anti, :prompts, :help, :clear
- Each task runs through 5 learning cycles
- Session summaries after each task

## Workflow

1. User provides task description via CLI or REPL
2. Agent reads current prompt from prompt.md
3. For each of 5 learning cycles:
   a. Inject memory.txt and anti-pattern.txt into prompt
   b. Execute claude CLI with enhanced prompt
   c. On success: extract learnings, update memory.txt
   d. On failure: extract anti-patterns, improve prompt
   e. Print cycle report with learnings and anti-patterns
4. Print session summary with all accumulated knowledge
5. In REPL mode: wait for next task

## Learning Cycle Flow

```
for cycle in 1..=5:
    enhanced_prompt = base_prompt + memory + anti_patterns + task
    result = run_agent(enhanced_prompt)

    if result.success:
        learnings = extract_learnings(result)
        save_to_memory(learnings)
        run_solution()
    else:
        anti_patterns = extract_anti_patterns(result.error)
        save_anti_patterns(anti_patterns)
        prompt = improve_prompt(prompt, error)

    print_cycle_report(cycle, learnings, anti_patterns)

print_session_summary()
```

## Cycle Report Format

```
============================================================
CYCLE N REPORT
============================================================
Status: SUCCESS/FAILED

Learnings acquired this cycle:
  + Learning 1
  + Learning 2

Anti-patterns identified this cycle:
  - Anti-pattern 1
  - Anti-pattern 2

Prompt was improved and archived for next cycle
============================================================
```

## Session Summary Format

```
############################################################
LEARNING SESSION SUMMARY
############################################################
Total cycles: 5
Successes: 3
Failures: 2

All learnings accumulated:
  + Learning 1
  + Learning 2
  + Learning 3

All anti-patterns identified:
  - Anti-pattern 1
  - Anti-pattern 2

Prompt versions created: 2
############################################################
```

## File Formats

### prompt.md
```markdown
# Current Prompt

<current system prompt here>

## Improvement from cycle 1 at 2024-01-01 12:00:00:
- Added improvement

# Past Prompts

## Version 1 - 2024-01-01 11:00:00
<archived prompt>
```

### memory.txt
```
- Always include error handling in generated code
- Use async/await for I/O operations
- File generation approach worked correctly
- Generated code executed without errors
```

### anti-pattern.txt
```
- Avoid hardcoded paths
- Do not use unwrap() without error context
- Avoid long-running operations without progress indicators
```

## CLI Interface

```bash
./run.sh "Create a REST API"
./run.sh --model opus "Build a CLI tool"
./run.sh --repl
./run.sh --list-prompts
./run.sh --show-memory
./run.sh --show-anti-patterns
```

## REPL Commands

```
agent> :help           # Show help
agent> :memory         # Show learnings
agent> :anti           # Show anti-patterns
agent> :prompts        # Show prompt history
agent> :clear          # Clear screen
agent> :quit           # Exit REPL
agent> Create a web server   # Start learning session
```

## Scripts

- build-all.sh: Compiles the Rust binary in release mode
- run.sh: Builds if needed and executes agent with provided task
- stop.sh: Kills any running agent-learner or claude processes
- test.sh: Runs cargo build, cargo test, and CLI validation

## Dependencies

- tokio: Async runtime
- serde/serde_json: Serialization
- uuid: Session identifiers
- chrono: Timestamps
- regex: Pattern matching

## Design Decisions

1. Single binary CLI: Matches prompt-2-k8s and multi-agent-verse patterns
2. File-based persistence: Simple, no database required
3. Claude CLI integration: Reuses existing infrastructure
4. 5 learning cycles: Each task runs 5 times to accumulate knowledge
5. Prompt versioning: Full history for analysis and rollback
6. REPL mode: Continuous interactive learning without restarting
7. Cycle-by-cycle reporting: Clear visibility into what was learned
8. Separate cycle directories: Each cycle output is isolated
9. Session summaries: Aggregated view of all learnings
10. No web frontend: Pure CLI tool for simplicity
