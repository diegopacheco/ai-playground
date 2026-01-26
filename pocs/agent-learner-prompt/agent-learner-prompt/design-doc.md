# Agent Learner Prompt - Design Document

## Overview

A self-learning CLI agent that iteratively improves its prompts based on execution results. The agent learns from successes and failures, storing knowledge in persistent files.

## Architecture

```
agent-learner-prompt/
├── src/
│   └── main.rs          # CLI entry point and orchestration
├── memory.txt           # Accumulated learnings
├── anti-pattern.txt     # Bad practices to avoid
├── prompt.md            # Prompt version history
├── solutions/           # Generated code output
├── run.sh               # Execute the agent
└── test.sh              # Validate functionality
```

## Core Components

### 1. Prompt Manager
- Reads current prompt from prompt.md
- Archives prompts to Past Prompts section before updates
- Generates improved prompts based on learnings

### 2. Memory System
- memory.txt: Stores successful patterns and learnings
- anti-pattern.txt: Stores failures and patterns to avoid
- Both files are injected into the system prompt

### 3. Agent Executor
- Spawns claude CLI as subprocess
- Captures stdout/stderr
- Implements retry loop with max 3 attempts
- Timeout of 300 seconds per attempt

### 4. Learning Extractor
- Analyzes agent output for success/failure patterns
- Extracts actionable learnings
- Updates memory.txt or anti-pattern.txt accordingly

### 5. Solution Runner
- Executes generated code in solutions/{project}/
- Validates output by running run.sh if present
- Reports success/failure back to learning system

## Workflow

1. User provides task description via CLI
2. Agent reads current prompt from prompt.md
3. Agent injects memory.txt and anti-pattern.txt into system prompt
4. Agent executes claude CLI with enhanced prompt
5. On success: extract learnings, update memory.txt
6. On failure: extract anti-patterns, retry with improved prompt
7. After max retries: archive current prompt, generate improved version
8. Run generated code in solutions/

## Retry Logic

```
attempt = 0
while attempt < MAX_RETRIES:
    result = run_agent(prompt)
    if result.success:
        extract_learnings(result)
        break
    else:
        extract_anti_patterns(result)
        prompt = improve_prompt(prompt, result.error)
        attempt += 1
```

## File Formats

### prompt.md
```markdown
# Current Prompt

<current system prompt here>

# Past Prompts

## Version 1 - 2024-01-01
<archived prompt>

## Version 2 - 2024-01-02
<archived prompt>
```

### memory.txt
```
- Always include error handling in generated code
- Use async/await for I/O operations
- Prefer explicit types over inference for public APIs
```

### anti-pattern.txt
```
- Avoid hardcoded paths
- Do not use unwrap() without error context
- Never skip input validation
```

## CLI Interface

```bash
./agent-learner "Create a REST API that returns hello world"
./agent-learner --list-prompts
./agent-learner --show-memory
./agent-learner --show-anti-patterns
```

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
4. Retry with learning: Each failure improves subsequent attempts
5. Prompt versioning: Full history for analysis and rollback
