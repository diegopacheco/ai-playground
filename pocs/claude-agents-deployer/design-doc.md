# Claude Code Agent Deployer - Design Document

## Overview

A Rust CLI tool that deploys markdown-based agent definitions to Claude Code as sub-agents.
The solution dynamically discovers agent files from an `agents/` folder and provides an interactive wizard for installation.

## Architecture

```
claude-agents-deployer/
├── agents/
│   ├── react-developer-agent.md
│   ├── rust-backend-developer-agent.md
│   ├── java-backend-developer-agent.md
│   ├── go-backend-developer-agent.md
│   ├── relational-dba-agent.md
│   ├── unit-testers-agent.md
│   ├── integration-tester-agent.md
│   ├── feature-documenter-agent.md
│   ├── design-doc-syncer-agent.md
│   ├── ui-testing-playwright-agent.md
│   ├── k6-stress-test-agent.md
│   ├── changes-sumarizer-agent.md
│   ├── code-reviewer-agent.md
│   └── security-reviewer-agent.md
├── skills/
│   ├── workflow-skill/
│   │   └── SKILL.md
│   └── workflow.md
├── src/
│   └── main.rs
├── Cargo.toml
└── design-doc.md
```

## Components

### 1. Agent Discovery Module
- Scans `agents/` folder for `.md` files
- Dynamically builds list of available agents
- Parses agent name from filename (e.g., `react-developer-agent.md` → "React Developer Agent")

### 2. Interactive Wizard
- Uses `dialoguer` crate for terminal UI
- Checkbox-based selection for agents
- Questions flow:
  1. Install all agents or select specific ones?
  2. If select: Show checkboxes for each discovered agent
  3. Install globally or locally?
  4. Turn selected agents into commands? (checkbox selection)

### 3. Installer
- Copies selected `.md` files to Claude Code installation directory
- Global: `~/.claude/agents/`
- Local: `./.claude/agents/`
- Creates command wrappers if requested

### 4. Command Generator
- Creates shell scripts/aliases for agents selected as commands
- Adds to `~/.claude/commands/` or `./.claude/commands/`

### 5. Workflow Installer
- Asks user if they want to install the workflow skill and command (default: yes)
- Copies `skills/workflow-skill/SKILL.md` to `{target}/skills/workflow-skill/SKILL.md`
- Copies `skills/workflow.md` as command `{target}/commands/ad/wf.md`
- Enables `/ad:wf` command that orchestrates all agents in a phased pipeline

## Technical Details

### Rust Version
- Edition: 2024
- Version: 1.83+ (latest stable supporting 2024 edition)

### Dependencies
```toml
[dependencies]
dialoguer = "0.11"
console = "0.15"
dirs = "5.0"
walkdir = "2.5"
```

### Installation Paths

| Type   | Agents Path          | Commands Path          | Skills Path                       |
|--------|----------------------|------------------------|-----------------------------------|
| Global | ~/.claude/agents/    | ~/.claude/commands/    | ~/.claude/skills/workflow-skill/  |
| Local  | ./.claude/agents/    | ./.claude/commands/    | ./.claude/skills/workflow-skill/  |

### Skill & Command Paths (relative in source)

| Source File                      | Installed As                                |
|----------------------------------|---------------------------------------------|
| skills/workflow-skill/SKILL.md   | {target}/skills/workflow-skill/SKILL.md      |
| skills/workflow.md               | {target}/commands/ad/wf.md                   |

### Wizard Flow

```
┌─────────────────────────────────────────────────────────┐
│         Claude Code Agent Deployer                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ? Install all agents or select specific ones?          │
│    ○ Install all agents                                 │
│    ● Select specific agents                             │
│                                                         │
│  ? Select agents to install:                            │
│    [x] React Developer Agent                            │
│    [x] Rust Backend Developer Agent                     │
│    [ ] Relational DBA Agent                             │
│    [x] Unit Testers Agent                               │
│    ...                                                  │
│                                                         │
│  ? Installation type:                                   │
│    ○ Global (~/.claude/)                                │
│    ● Local (./.claude/)                                 │
│                                                         │
│  ? Turn agents into commands?                           │
│    [x] React Developer Agent                            │
│    [ ] Rust Backend Developer Agent                     │
│    [x] Unit Testers Agent                               │
│    ...                                                  │
│                                                         │
│  ? Install workflow skill/command (/ad:wf)? [Y/n]       │
│                                                         │
│  Installed 3 agents                                     │
│  Created 2 commands                                     │
│  Installed workflow skill and /ad:wf command             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Agent File Format

Each agent `.md` file contains:
- Agent name and description
- System prompt/instructions
- Capabilities and tools available
- Constraints and guidelines

## Dynamic Behavior

- Adding new `.md` files to `agents/` folder automatically includes them in wizard
- Removing `.md` files removes them from wizard options
- No code changes required when agents are added/removed

## Build & Run

```bash
cargo build --release
./target/release/claude-agents-deployer
```

## Success Criteria

1. Binary discovers all `.md` files in `agents/` folder
2. Interactive wizard with checkbox selection works
3. Agents install to correct paths (global/local)
4. Commands are created when requested
5. Solution is fully dynamic (no hardcoded agent list)
6. Workflow skill and `/ad:wf` command install when requested (default: yes)
