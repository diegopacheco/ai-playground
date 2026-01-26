# Agent Learner Prompt

A self-learning CLI agent that iteratively improves its prompts based on execution results.

## Features

- Learns from successes (memory.txt)
- Avoids known failures (anti-pattern.txt)
- Maintains prompt version history (prompt.md)
- Retry loop with max 3 attempts
- Runs generated code in solutions/

## Build

```bash
cargo build --release
```

## Usage

```bash
./run.sh "Create a hello world web server in Python"
./run.sh --model opus "Build a REST API"
./run.sh --show-memory
./run.sh --show-anti-patterns
./run.sh --list-prompts
./run.sh --help
```

## Test

```bash
./test.sh
```

## Files

- `memory.txt` - Accumulated learnings from successful runs
- `anti-pattern.txt` - Patterns to avoid from failures
- `prompt.md` - Current and past prompt versions
- `solutions/` - Generated code output directory
