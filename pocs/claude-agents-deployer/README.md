# Claude Agents Deployer

A Rust CLI tool that deploys markdown-based agent definitions to Claude Code as sub-agents.

## What It Does

Claude Agents Deployer provides an interactive wizard to install pre-built AI agents into your Claude Code environment. It dynamically discovers agent files from the `agents/` folder and lets you choose which ones to install globally or locally. You can also turn agents into slash commands for quick access.

## Agents

| Agent | Description |
|-------|-------------|
| React Developer | Expert in modern React with hooks, TypeScript, and best practices |
| Rust Backend Developer | Builds high-performance services with Axum, Tokio, and async Rust |
| Java Backend Developer | Enterprise Java with Spring Boot, JPA, and microservices |
| Go Backend Developer | Scalable distributed systems with goroutines and channels |
| Relational DBA | Database design, optimization, and SQL expertise |
| Unit Testers | Comprehensive unit testing with mocks and high coverage |
| Integration Tester | End-to-end testing with real dependencies and Testcontainers |
| Feature Documenter | Technical writing for features, APIs, and user guides |
| Design Doc Syncer | Keeps design documents synchronized with implementation |
| UI Testing Playwright | Browser automation and visual regression testing |
| K6 Stress Test | Performance and load testing with k6 scripts |
| Changes Summarizer | Git diff analysis and release notes generation |
| Code Reviewer | Code quality review focusing on bugs and best practices |
| Security Reviewer | Security vulnerability detection and OWASP compliance |

## How It Works

1. **Discovery** - Scans the `agents/` folder for `.md` files
2. **Selection** - Interactive wizard lets you pick agents to install
3. **Installation** - Copies agent files to Claude Code directory
4. **Commands** - Optionally creates slash commands for quick agent access

## Installation Paths

| Type | Agents | Commands |
|------|--------|----------|
| Global | `~/.claude/agents/` | `~/.claude/commands/` |
| Local | `./.claude/agents/` | `./.claude/commands/` |

## Usage

```bash
./run.sh
```

Or build and run manually:

```bash
cargo build --release
./target/release/claude-agents-deployer
```

## Wizard Flow

```
? Install all agents? (y/n)
? Select agents to install: [space to toggle]
? Installation type: Global / Local
? Generate commands for agents? (y/n)
? Select agents to turn into commands: [space to toggle]
```

## Adding New Agents

Drop a new `.md` file in the `agents/` folder. The wizard automatically picks it up - no code changes required.

## Requirements

- Rust 1.93+ (2024 edition)

## Tool in Action

```
  Claude Code Agent Deployer

Found 15 agents:

  - Changes Sumarizer
  - Code Reviewer
  - Design Doc Syncer
  - Design Docer Syncer
  - Feature Documenter
  - Go Backend Developer
  - Integration Tester
  - Java Backend Developer
  - K6 Stress Test
  - React Developer
  - Relational Dba
  - Rust Backend Developer
  - Security Reviewer
  - Ui Testing Playwright
  - Unit Testers

✔ Install all agents? · yes
✔ Installation type · Local (./.claude/)
✔ Generate commands for agents? · yes
✔ Select agents to turn into commands (space to toggle, enter to confirm) · Changes Sumarizer, Code Reviewer, Design Doc Syncer, Design Docer Syncer, Feature Documenter, Go Backend Developer, Integration Tester, Java Backend Developer, K6 Stress Test, React Developer, Relational Dba, Rust Backend Developer, Security Reviewer, Ui Testing Playwright, Unit Testers

Installing agents...

  Installed: Changes Sumarizer
  Created command: /changes-sumarizer-agent
  Installed: Code Reviewer
  Created command: /code-reviewer-agent
  Installed: Design Doc Syncer
  Created command: /design-doc-syncer-agent
  Installed: Design Docer Syncer
  Created command: /design-docer-syncer-agent
  Installed: Feature Documenter
  Created command: /feature-documenter-agent
  Installed: Go Backend Developer
  Created command: /go-backend-developer-agent
  Installed: Integration Tester
  Created command: /integration-tester-agent
  Installed: Java Backend Developer
  Created command: /java-backend-developer-agent
  Installed: K6 Stress Test
  Created command: /k6-stress-test-agent
  Installed: React Developer
  Created command: /react-developer-agent
  Installed: Relational Dba
  Created command: /relational-dba-agent
  Installed: Rust Backend Developer
  Created command: /rust-backend-developer-agent
  Installed: Security Reviewer
  Created command: /security-reviewer-agent
  Installed: Ui Testing Playwright
  Created command: /ui-testing-playwright-agent
  Installed: Unit Testers
  Created command: /unit-testers-agent

Done!
  Installed 15 agents to ".claude/agents"
  Created 15 commands in ".claude/commands"
```