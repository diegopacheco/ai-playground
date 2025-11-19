# Claude Code POCs

* `custom-agents/` contains security-focused custom agents for automated security scanning secrets.

* `claude-skill-fun/` demonstrates JSON formatting, validation, and minification skills with zero external dependencies.

* `custom-command-fun/` provides slash commands like `/generate-tests` for automated unit test generation.

* `custom-hook/` implements code quality hooks that run linters and validators on file edits and tool executions.

## Decision Criteria

### A) When to use Prompts
Use prompts for one-time instructions or queries that don't require automation. Best for ad-hoc requests, exploratory questions, or when you need Claude to perform a single specific task without reusability requirements.

### B) When to use Custom Agents
Use custom agents for complex, multi-step automated workflows that run autonomously. Best for security scanning, code analysis, performance profiling, or any task requiring multiple tool invocations and systematic codebase exploration.

### C) When to use Custom Commands
Use custom commands for frequently repeated tasks that need quick invocation via slash syntax. Best for generating tests, documentation, or any workflow you execute regularly and want to trigger with `/command-name`.

### D) When to use Hooks
Use hooks for validation, enforcement, or automation that triggers on specific events. Best for running linters after edits, checking secrets before commits, enforcing code standards, or integrating external tools into your workflow automatically.

### E) When to use Claude Skills
Use skills for specialized, reusable capabilities that Claude auto-invokes based on intent. Best for file format operations (JSON, YAML), data transformations, or domain-specific tasks that Claude should detect and activate automatically without explicit commands.

### F) When to use MCP
Use MCP (Model Context Protocol) for integrating external data sources and services into Claude. Best for connecting databases, APIs, file systems, or third-party tools that Claude needs to access dynamically during conversations.
