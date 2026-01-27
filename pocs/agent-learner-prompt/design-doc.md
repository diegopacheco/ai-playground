# Agent Learner Prompt - Design Document

## Overview

A self-learning CLI agent that iteratively improves its prompts based on execution results. The agent learns from successes and failures, storing knowledge in persistent files per project. Runs 3 learning cycles per task (configurable) with code review and supports interactive REPL mode.

## Rust Guidelines

* Have modules which are folder
* Each module has mod.rs
* mod.rs should only expose public functions/types - should not have functions or logic itself here.
* Make sure there is at least 1 file per module besides mod.rs
* Use `anyhow` for error handling in main logic
* Prefer returning `Result<T, anyhow::Error>` from functions
* Use pattern matching for error handling where appropriate
* Keep functions small and focused - single responsibility
* Use `clap` for CLI argument parsing
* Use `tokio` for async operations
* Write unit tests for core logic

## Architecture

```
agent-learner-prompt/
├── src/
│   ├── main.rs              # CLI entry point and orchestration
│   └── agents/              # Agent wrappers
│       ├── mod.rs           # Agent dispatcher
│       ├── claude.rs        # Claude CLI wrapper
│       ├── codex.rs         # Codex CLI wrapper
│       ├── copilot.rs       # Copilot CLI wrapper
│       └── gemini.rs        # Gemini CLI wrapper
├── solutions/               # Generated code output
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
├── build-all.sh             # Build the project
├── run.sh                   # Execute the agent
├── stop.sh                  # Stop running agents
└── test.sh                  # Validate functionality
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
- Spawns selected CLI agent as subprocess
- Displays current agent and model at startup and each cycle
- Captures stdout/stderr
- Implements configurable learning cycles (default: 3)
- Timeout of 300 seconds per agent call

### 4. Agent Wrappers

Separate wrapper modules for each supported CLI agent:

| Agent | CLI Command | Model Flag | Default Model |
|-------|-------------|------------|---------------|
| **claude** | `claude -p <prompt> --model <model> --dangerously-skip-permissions` | `--model` | sonnet |
| **codex** | `codex exec --full-auto --model <model> <prompt>` | `--model` | gpt-5.2 |
| **copilot** | `copilot --allow-all --model <model> -p <prompt>` | `--model` | claude-sonnet-4 |
| **gemini** | `gemini -y <prompt>` | (none) | gemini-2.5-pro |

Each wrapper:
- Spawns the CLI as subprocess
- Captures stdout/stderr
- Writes logs to cycle directory
- Returns success/failure status

### 5. Model Selector
- Switch agent via `--agent` flag or `:agent` command in REPL
- Switch model via `--model` flag or `:model` command in REPL
- Displays current agent and model prominently
- Available agents: claude, codex, copilot, gemini
- Shows agent/model in cycle headers

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
- **Phase 4**: Asks the LLM to extract learnings from the cycle
- **Phase 5**: Asks the LLM to identify mistakes to avoid
- Prompt templates ask LLM to analyze code and execution results
- Saves responses to memory.txt and mistakes.txt
- Filters duplicates before saving

### 8. Prompt Improver
- **Phase 6**: Asks the LLM to suggest an improved prompt
- Provides current prompt, learnings, and mistakes as context
- LLM generates better prompt for next cycle
- Archives old prompt and updates prompts.md

### 9. Final Code Generator
- After all cycles complete, copies best result to code/ folder
- **Recursively copies all files and subdirectories**
- code/ contains the final production-ready output
- Clean folder without cycle artifacts (prompt.txt, output.txt, review.txt, learnings.txt, mistakes.txt, improved_prompt.txt)

### 10. REPL Mode
- Interactive loop for continuous learning
- Commands: :quit, :cycles, :agent, :model, :memory, :mistakes, :prompts, :help, :clear
- Configurable cycles, agent, and model per session
- Session summaries after each task

## Workflow

1. User provides task description via CLI or REPL
2. Agent creates project folder in solutions/{project}/
3. Initialize prompts.md with default prompt (if not exists)
4. For each learning cycle (default 3):
   a. Display current agent and model being used
   b. **Phase 1**: Execute selected agent with enhanced prompt (from prompts.md + memory.txt + mistakes.txt)
   c. **Phase 2**: Run solution with 10s timeout
   d. **Phase 3**: Review code for architecture/design/security/tests
   e. **Phase 4**: Ask LLM to extract learnings from this cycle
   f. **Phase 5**: Ask LLM to identify mistakes to avoid
   g. **Phase 6**: Ask LLM to suggest improved prompt
   h. Update memory.txt with LLM learnings
   i. Update mistakes.txt with LLM mistakes
   j. Archive current prompt and update prompts.md with LLM suggestion
   k. Print cycle report
5. Copy final successful cycle to code/ folder (recursive)
6. Print session summary with all accumulated knowledge
7. In REPL mode: wait for next task

## Learning Cycle Phases

```
project_dir = solutions/{project}/
init prompts.md, memory.txt, mistakes.txt in project_dir

for cycle in 1..=num_cycles:
    print "Using agent: {agent} | model: {model}"
    
    Phase 1: Generate Code
        enhanced_prompt = prompts.md + memory.txt + mistakes.txt + task
        result = run_agent(agent, enhanced_prompt, model)
        save to cycle-{n}/

    Phase 2: Run Solution (with 10s timeout)
        run_solution_with_timeout()
        if timeout: treat as success (web server)

    Phase 3: Code Review
        review_prompt = check architecture, design, security, tests
        findings = run_agent(agent, review_prompt, model)
        parse findings into categories

    Phase 4: Extract Learnings (LLM)
        learnings_prompt = "What worked well? What learnings from this cycle?"
        learnings = run_agent(agent, learnings_prompt, model)
        save to memory.txt

    Phase 5: Extract Mistakes (LLM)
        mistakes_prompt = "What mistakes should be avoided? What went wrong?"
        mistakes = run_agent(agent, mistakes_prompt, model)
        save to mistakes.txt

    Phase 6: Improve Prompt (LLM)
        improve_prompt = "Given learnings and mistakes, suggest improved prompt"
        new_prompt = run_agent(agent, improve_prompt, model)
        archive old prompt and update prompts.md

    print_cycle_report()

copy last successful cycle to code/ (recursive)
print_session_summary()
```

## Project Folder Structure

Each project in solutions/ has:
```
solutions/{project}/
├── memory.txt       # Learnings accumulated across cycles
├── mistakes.txt     # Mistakes to avoid (fed to each cycle)
├── prompts.md       # Prompt versions (updated EVERY cycle)
├── cycle-1/         # First attempt
│   ├── prompt.txt   # Prompt used this cycle
│   ├── output.txt   # Agent output
│   ├── review.txt   # Code review results
│   ├── learnings.txt    # LLM-extracted learnings
│   ├── mistakes.txt     # LLM-extracted mistakes
│   ├── improved_prompt.txt  # LLM-suggested improved prompt
│   └── (generated code files and subdirectories)
├── cycle-2/         # Second attempt (improved)
├── cycle-3/         # Third attempt (improved)
└── code/            # Final production code (recursive copy, clean)
```

## Prompt Update Flow

prompts.md is updated after EVERY cycle:
1. Read current prompt from prompts.md
2. Build enhanced prompt (current + memory + mistakes + task)
3. Execute cycle
4. Review code
5. Archive current prompt to "Past Prompts" section
6. Generate improved prompt based on findings
7. Write new prompt to "Current Prompt" section
8. Next cycle uses improved prompt

## Review Output Format

The review phase asks the LLM to output:
```
ARCHITECTURE: <issues or OK>
DESIGN: <issues or OK>
CODE_QUALITY: <issues or OK>
SECURITY: <issues or OK>
TESTS: <issues or OK>
```

## LLM Learning Prompts

### Phase 4: Extract Learnings
```
Analyze the code generation cycle that just completed. The task was: {task}

Review the generated code and execution results:
- Code output: {output}
- Solution run result: {solution_result}
- Review findings: {review_summary}

What specific, actionable learnings can be extracted from this cycle?
Focus on patterns that worked well and should be repeated.
Do NOT include generic statements like "task completed successfully".

Output format (one learning per line, be specific):
LEARNING: <specific actionable insight>
LEARNING: <specific actionable insight>
```

### Phase 5: Extract Mistakes
```
Analyze the code generation cycle for mistakes to avoid. The task was: {task}

Review the generated code and execution results:
- Code output: {output}
- Solution run result: {solution_result}
- Review findings: {review_summary}

What specific mistakes were made that should be avoided in future cycles?
Focus on concrete anti-patterns, not generic advice.

Output format (one mistake per line, be specific):
MISTAKE: <specific mistake to avoid>
MISTAKE: <specific mistake to avoid>
```

### Phase 6: Improve Prompt
```
You are a prompt engineer. Your task is to improve the code generation prompt.

Current prompt:
{current_prompt}

Learnings from this cycle:
{learnings}

Mistakes identified:
{mistakes}

Review findings:
{review_summary}

Generate an improved version of the prompt that:
1. Incorporates the learnings as guidelines
2. Explicitly warns against the identified mistakes
3. Addresses the review findings
4. Remains clear and actionable

Output ONLY the improved prompt text, no explanations.
```

## Filtered Learnings

These generic learnings are filtered out:
- "Task completed successfully"
- "Generated code executed"
- "File generation approach worked"
- "Code produced valid output"

Only specific learnings are saved to memory.txt:
- "Tests passed for task: create web server"
- "Build succeeded without errors"
- "Code passed linting checks"
- "Architecture passed review - structure is appropriate"
- "Security passed review - no vulnerabilities found"

## Files Renamed

| Old Name | New Name | Location |
|----------|----------|----------|
| anti-pattern.txt | mistakes.txt | solutions/{project}/ |
| prompt.md | prompts.md | solutions/{project}/ |
| memory.txt | memory.txt | solutions/{project}/ |

Note: These files no longer exist in project root. Each project has its own isolated learning context.

## CLI Interface

```bash
./run.sh "Create a REST API"
./run.sh --cycles 5 "Build a CLI tool"
./run.sh --agent claude --model opus "Quick task"
./run.sh --agent codex --model gpt-5.2 "Build API"
./run.sh --agent copilot "Use copilot agent"
./run.sh --agent gemini "Use gemini agent"
./run.sh --repl
```

## Agent and Model Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--agent <name>` | `-a` | Select CLI agent (claude, codex, copilot, gemini) |
| `--model <name>` | `-m` | Select model for the agent |
| `--cycles <n>` | `-c` | Number of learning cycles (1-10) |
| `--repl` | | Enter interactive REPL mode |

## Supported Agents and Models

### Claude (default)
```bash
./run.sh --agent claude --model sonnet "task"
./run.sh --agent claude --model opus "task"
./run.sh --agent claude --model haiku "task"
```

### Codex (OpenAI)
```bash
./run.sh --agent codex --model gpt-5.2 "task"
./run.sh --agent codex --model gpt-5.1 "task"
./run.sh --agent codex --model gpt-4.1 "task"
```

### Copilot (GitHub)
```bash
./run.sh --agent copilot --model claude-sonnet-4 "task"
./run.sh --agent copilot --model gpt-5.1 "task"
```

### Gemini (Google)
```bash
./run.sh --agent gemini "task"
```

## Model Display

At startup and each cycle:
```
============================================================
Agent Learner Starting...
Task: Create a REST API
Agent: claude (use --agent to change)
Model: sonnet (use --model to change)
Learning cycles: 3
============================================================

************************************************************
LEARNING CYCLE 1/3 [Agent: claude | Model: sonnet]
************************************************************
```

## REPL Commands

```
agent> :help           # Show help
agent> :agent claude   # Switch to claude agent
agent> :agent codex    # Switch to codex agent
agent> :agent copilot  # Switch to copilot agent
agent> :agent gemini   # Switch to gemini agent
agent> :agent          # Show current agent
agent> :model opus     # Switch to opus model
agent> :model          # Show current model
agent> :cycles 5       # Set cycles to 5
agent> :cycles         # Show current cycles
agent> :memory         # Show learnings (from current project)
agent> :mistakes       # Show mistakes to avoid
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
- stop.sh: Kills any running agent-learner or agent CLI processes
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
3. **Multi-agent support**: Wrappers for claude, codex, copilot, gemini
4. **Easy agent switching**: --agent flag and :agent REPL command
5. **Easy model switching**: --model flag and :model REPL command
6. **Code review phase**: Each cycle reviews architecture, design, security, tests
7. **Filter generic learnings**: Only save specific, actionable insights
8. **10s solution timeout**: Prevents blocking on web servers
9. **Timeout = success for servers**: Web apps that start are considered working
10. **Per-project learning files**: memory.txt, mistakes.txt, prompts.md in solutions/{project}/
11. **prompts.md updated every cycle**: Continuous improvement, not just on failure
12. **code/ folder**: Final clean output separate from cycle artifacts
13. **Agent/model display**: Show agent and model at startup and each cycle header
14. **REPL mode**: Continuous interactive learning with agent/model switching
