# Agent Learner Prompt - Design Document

## Overview

A self-learning CLI agent that iteratively improves its prompts based on execution results. The agent learns from successes and failures, storing knowledge in persistent files per project. Runs 3 learning cycles per task (configurable) with code review and supports interactive REPL mode.

## Architecture

```
agent-learner-prompt/
├── src/
│   └── main.rs          # CLI entry point and orchestration
├── solutions/           # Generated code output
│   └── {project}/
│       ├── memory.txt       # Project-specific learnings
│       ├── mistakes.txt     # Project-specific mistakes to avoid
│       ├── prompts.md       # Prompt version history (updated each cycle)
│       ├── cycle-1/         # First cycle output
│       │   ├── prompt.txt
│       │   ├── output.txt
│       │   └── review.txt
│       ├── cycle-2/         # Second cycle output
│       ├── cycle-3/         # Third cycle output
│       └── code/            # Final code (copy of last successful cycle)
├── build-all.sh         # Build the project
├── run.sh               # Execute the agent
├── stop.sh              # Stop running agents
└── test.sh              # Validate functionality
```

## Core Components

### 1. Prompt Manager
- Reads current prompt from prompts.md (per project)
- Archives prompts to Past Prompts section before updates
- Updates prompts.md after EVERY cycle with improvements
- Generates improved prompts based on learnings and review findings
- Timestamps each prompt version

### 2. Memory System
- memory.txt: Stores successful patterns and learnings (per project)
- mistakes.txt: Stores failures and patterns to avoid (per project)
- All learning files live in solutions/{project}/ folder
- Filters out generic/vanilla learnings
- Only saves specific, actionable insights
- Each cycle reads and updates these files

### 3. Agent Executor
- Spawns claude CLI as subprocess
- Displays current model being used at startup and each cycle
- Captures stdout/stderr
- Implements configurable learning cycles (default: 3)
- Timeout of 300 seconds per agent call

### 4. Model Selector
- Displays current model prominently
- Switch via --model flag or :model command in REPL
- Supported models: sonnet, opus, haiku
- Shows model in cycle headers

### 5. Code Review Phase
Each successful cycle includes a review phase that checks:
- **Architecture**: Is the structure appropriate?
- **Design**: Are patterns used correctly?
- **Code Quality**: Any bad practices or code smells?
- **Security**: Any vulnerabilities (injection, XSS, hardcoded secrets)?
- **Tests**: Are there tests? Good coverage?

### 6. Solution Runner
- Executes generated code with 10 second timeout
- Timeout treated as success (likely a web server running)
- Captures output for analysis

### 7. Learning Extractor
- Filters out generic learnings like "Task completed successfully"
- Extracts specific learnings: tests passed, build succeeded, lint passed
- Converts review findings to actionable insights
- Updates memory.txt or mistakes.txt in project folder

### 8. Final Code Generator
- After all cycles complete, copies best result to code/ folder
- code/ contains the final production-ready output
- Clean folder without cycle artifacts (prompt.txt, output.txt, review.txt)

### 9. REPL Mode
- Interactive loop for continuous learning
- Commands: :quit, :cycles, :model, :memory, :mistakes, :prompts, :help, :clear
- Configurable cycles and model per session
- Session summaries after each task

## Workflow

1. User provides task description via CLI or REPL
2. Agent reads current prompt from prompt.md
3. For each learning cycle (default 3):
   a. **Phase 1**: Execute agent with enhanced prompt
   b. **Phase 2**: Run solution with 10s timeout
   c. **Phase 3**: Review code for architecture/design/security/tests
   d. Extract learnings and anti-patterns from review
   e. Print cycle report
4. Print session summary with all accumulated knowledge
5. In REPL mode: wait for next task

## Learning Cycle Phases

```
for cycle in 1..=num_cycles:
    Phase 1: Generate Code
        enhanced_prompt = base_prompt + memory + anti_patterns + task
        result = run_agent(enhanced_prompt)

    Phase 2: Run Solution (with 10s timeout)
        run_solution_with_timeout()
        if timeout: treat as success (web server)

    Phase 3: Code Review
        review_prompt = check architecture, design, security, tests
        findings = run_agent(review_prompt)
        parse findings into categories

    Extract learnings (filter generic ones)
    Extract anti-patterns from issues found
    Improve prompt if issues detected

    print_cycle_report()

print_session_summary()
```

## Review Output Format

The review phase asks the LLM to output:
```
ARCHITECTURE: <issues or OK>
DESIGN: <issues or OK>
CODE_QUALITY: <issues or OK>
SECURITY: <issues or OK>
TESTS: <issues or OK>
```

## Filtered Learnings

These generic learnings are filtered out:
- "Task completed successfully"
- "Generated code executed"
- "File generation approach worked"
- "Code produced valid output"

Only specific learnings are saved:
- "Tests passed for task: create web server"
- "Build succeeded without errors"
- "Code passed linting checks"
- "Architecture passed review - structure is appropriate"
- "Security passed review - no vulnerabilities found"

## CLI Interface

```bash
./run.sh "Create a REST API"
./run.sh --cycles 5 "Build a CLI tool"
./run.sh --model opus --cycles 2 "Quick task"
./run.sh --repl
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
```

## Timeouts

- Agent execution: 300 seconds (5 minutes)
- Solution run: 10 seconds (web servers timeout = OK)

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

1. **3 cycles default**: Faster iteration than 5, still enough to learn
2. **Configurable cycles**: --cycles N or :cycles N in REPL
3. **Code review phase**: Each cycle reviews architecture, design, security, tests
4. **Filter generic learnings**: Only save specific, actionable insights
5. **10s solution timeout**: Prevents blocking on web servers
6. **Timeout = success for servers**: Web apps that start are considered working
7. **File-based persistence**: Simple, no database required
8. **Claude CLI integration**: Reuses existing infrastructure
9. **REPL mode**: Continuous interactive learning
10. **Cycle-by-cycle reporting**: Clear visibility into what was learned
