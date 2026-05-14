# Agent Harness Patterns

A catalog of recurring patterns observed across modern agent harnesses (Claude Code, Codex, Cursor, Gemini CLI, Augment, Aider, OpenCode, BMAD, SuperClaude, etc.). These are the building blocks that make a coding agent useful, controllable, and extensible.

## Catalog

### 1. Progressive Disclosure
Reveal capabilities, instructions, and context incrementally rather than dumping everything up front. The harness exposes a small surface by default; deeper knowledge (skills, sub-agents, tool schemas, plugin docs) is loaded only when the model signals intent.
- **Examples**: Claude Code Skills (loaded on `/skill-name`), deferred tool schemas (`ToolSearch`), lazy MCP server attachment, on-demand `CLAUDE.md` reads.
- **Why it matters**: Keeps the context window cheap; avoids polluting reasoning with irrelevant noise.

### 2. Advisor Pattern
A sub-agent or role gives recommendations without holding the keys to mutate state. The main agent decides whether to act on the advice.
- **Examples**: `Plan` agent in Claude Code, `code-reviewer-agent`, `security-reviewer-agent`, BMAD analyst/architect roles, Aider's `--ask` mode.
- **Why it matters**: Separates judgment from execution; lets you parallelize "second opinions" without risking writes.

### 3. Escape Hatch
A deliberate way for the user to break out of a constrained mode, override safety, or hand control back to a human.
- **Examples**: `ExitPlanMode`, `dangerouslyDisableSandbox`, `--no-verify` (with explicit user consent), `Other` option in `AskUserQuestion`, manual `! <command>` prefix.
- **Why it matters**: Constraints are useful only if they are not prisons; users need a clear way out without fighting the harness.

### 4. Include / Local Config Files
Hierarchical config picked up automatically from the filesystem — global → user → project → directory. Allows policy and context to travel with the codebase.
- **Examples**: `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.aider.conf.yml`, `GEMINI.md`, `.augmentignore`, `settings.json` / `settings.local.json` layering.
- **Why it matters**: Lets teams encode conventions once and have every agent invocation respect them automatically.

### 5. Generic Execution Engine (Harness Core)
The harness itself is a thin, language-agnostic loop: read prompt → call model → execute tools → feed results back. All domain knowledge lives outside the engine.
- **Examples**: Claude Code's tool-loop, Codex CLI core, Aider's edit loop, OpenCode runtime, the Anthropic Agent SDK.
- **Why it matters**: A small, stable core is easier to harden, audit, and port across models; specialization is pushed to data (skills, configs, prompts).

### 6. Specialized Rules (Per-Task / Per-Language Files)
Reusable instruction packs scoped to a task type, language, or framework, selected at invocation time.
- **Examples**: Claude Code Skills (`java-backend-developer-agent`, `react-developer-agent`, `k6-stress-test-agent`), Cursor rule files, BMAD method packs, SuperClaude personas.
- **Why it matters**: Avoids re-explaining the same conventions every session; lets domain experts ship "best-practice bundles" as artifacts.

### 7. Hooks (PreTool / PostTool / Stop / SubmitPrompt)
Shell commands that run on lifecycle events, letting users inject linting, telemetry, redaction, or routing without modifying the agent.
- **Examples**: Claude Code `PreToolUse`/`PostToolUse`/`UserPromptSubmit` hooks, `cc-hook-tool-time-tracker`, context-mode auto-routing hook.
- **Why it matters**: Deterministic, user-owned guardrails outside the model's control plane.

### 8. Tool Allowlist / Permission Mode
Coarse and fine-grained gating of which tools the agent can call without confirmation. Modes include `plan`, `acceptEdits`, `bypassPermissions`, `default`.
- **Examples**: Claude Code permission modes, Cursor "auto-accept", Aider `--yes`, Codex `--dangerously-bypass`.
- **Why it matters**: Lets the same agent run interactively, semi-autonomously, or in CI without code changes.

### 9. Plan Mode / Dry-Run Separation
A read-only thinking phase that produces a plan; execution is a distinct, approval-gated phase.
- **Examples**: Claude Code Plan mode + `ExitPlanMode`, Aider `/architect`, BMAD's plan→build split, SDD (Spec-Driven Development) flows.
- **Why it matters**: Cheap to iterate on intent before paying the cost of edits and side effects.

### 10. Sub-Agents / Delegation
Spawn isolated agent instances for sub-tasks; each has its own context window and tool set.
- **Examples**: Claude Code `Agent` tool, BMAD sub-roles, CrewAI crews, Strands multi-agents, Conductor.
- **Why it matters**: Protects the main context from large research outputs; enables parallel work.

### 11. Memory / Persistence Layer
Durable, file-based notes that survive across sessions — user profile, feedback, project state, references.
- **Examples**: Claude Code's `auto memory` (`MEMORY.md` + per-fact files), Aider chat history, Open Memory, Strands memory, beads.
- **Why it matters**: The agent stops re-asking the same questions and accumulates a working model of the user and project.

### 12. Slash Commands / Skills
User-invocable, named workflows that bundle a prompt template, allowed tools, and sometimes scripts.
- **Examples**: Claude Code slash commands, Cursor commands, BMAD "agents", custom skills like `/ultrareview`, `/security-review`, `/autobench`.
- **Why it matters**: Turns repeatable workflows into first-class, shareable artifacts.

### 13. MCP (Model Context Protocol) / External Tool Integration
A standardized protocol for plugging external tool servers (browsers, databases, repos, drives) into any harness.
- **Examples**: `mcp__playwright__*`, `mcp__repo-mcp__*`, Google Drive MCP, context-mode plugin.
- **Why it matters**: Decouples tool authors from harness authors; one MCP server works across Claude, Cursor, Codex, etc.

### 14. Context Compaction / Summarization
Automatically compress older conversation turns when the context window fills, keeping the recent tail verbatim.
- **Examples**: Claude Code auto-compaction, Aider `--map-tokens`, Cursor's rolling summary.
- **Why it matters**: Enables long sessions without losing the thread; trades fidelity for continuity.

### 15. Worktree / Sandbox Isolation
Run risky operations in an isolated git worktree, container, or sandbox so the main checkout is untouched until merge.
- **Examples**: Claude Code `isolation: "worktree"`, Codex sandbox, Cursor background agents, Anthropic Managed Agents.
- **Why it matters**: Lets the agent move fast without endangering uncommitted work.

### 16. Background / Async Tasks
Long-running tool calls or agents that don't block the main loop; the harness notifies the model when they finish.
- **Examples**: `run_in_background` on `Bash` and `Agent`, `Monitor` for streaming output, Continuous Claude, Cook (Race/Review/Orchestrate loops).
- **Why it matters**: Frees the agent to keep working while CI, deploys, or peer reviews run.

### 17. Scheduling / Cron / Loops
Run a prompt or skill on a schedule or until a condition is met.
- **Examples**: `/loop`, `/schedule`, `CronCreate`, `ScheduleWakeup`, Continuous Claude.
- **Why it matters**: Turns the agent into a stand-alone operator for monitoring, polling, and recurring chores.

### 18. Task / TODO Tracking
A structured to-do list the agent maintains as it works; visible to the user and survives context compaction.
- **Examples**: `TaskCreate`/`TaskUpdate`, `TodoWrite`, beads, SuperClaude task tracker.
- **Why it matters**: Externalizes planning state so neither agent nor user loses track of multi-step work.

### 19. Just-in-Time Tool Loading
The full tool catalog is referenced by name only; schemas are fetched on demand via a search tool.
- **Examples**: `ToolSearch` with `select:` queries, MCP lazy mounting, Skill-gated tool unlocks.
- **Why it matters**: Hundreds of tools become viable without burning tokens on schemas the agent never calls.

### 20. Confirmation Gates / Risk-Aware Prompts
The harness asks the user before destructive, irreversible, or shared-state actions (force-push, `rm -rf`, sending messages, dropping tables).
- **Examples**: Claude Code's "executing actions with care" policy, Cursor's destructive-command modal, Aider's git-commit prompts.
- **Why it matters**: Blast-radius awareness baked into the loop, not left to the model's discretion alone.

### 21. Multi-Agent Orchestration
Multiple specialized agents coordinated by a router, judge, or orchestrator — sometimes adversarial.
- **Examples**: Multi-Agent Verse, Auction House, Werewolf, Debate Club, PR Agent, Pixel Office, BMAD method, CrewAI, Strands.
- **Why it matters**: Different roles, models, or temperatures excel at different sub-problems; orchestration composes them.

### 22. Read-Before-Write Contract
The harness enforces that the model has read a file (or holds a fresh snapshot) before editing it.
- **Examples**: Claude Code's `Edit` requires prior `Read`, Aider's edit-block format, Cursor's apply-after-show.
- **Why it matters**: Prevents stale-overwrite bugs and forces the agent to ground its edits in current state.

### 23. Status Line / Observability
A persistent, model-visible (or user-visible) status surface — current branch, model, token usage, costs, queue.
- **Examples**: Claude Code status line skill, OpenTelemetry agent traces, Agent Observability stack, prompt-score, semantic drift detector.
- **Why it matters**: Makes the agent's resource usage and decisions inspectable instead of opaque.

### 24. Self-Correction / Retry Loops
When a tool call fails (tests red, lint error, hook block), the agent is expected to diagnose and fix rather than escalate.
- **Examples**: Aider's auto-lint/auto-test loop, Claude Code's "fix the underlying issue" guidance, Ralph reflection loops, Lisa Loop (agent-learner-prompt).
- **Why it matters**: Turns brittle one-shot generation into a robust converging process.

### 25. Spec-Driven / Test-First Flows
Force the agent to write a spec, design doc, or test before code; subsequent edits are gated by the spec.
- **Examples**: Verified SDD (VSDD), SDD-research, BMAD's PRD-first flow, autobench skill.
- **Why it matters**: Reduces hallucinated requirements; gives a checkable artifact to review against.

### 26. Prompt / Skill Composition
Skills, personas, and prompts can be layered, included, or chained — like CSS cascades for instructions.
- **Examples**: SuperClaude personas, Skill stacking in Claude Code, BMAD method packs, Cursor rule includes.
- **Why it matters**: DRY for prompts; teams build a library instead of copy-pasting mega-prompts.

### 27. Snapshot / Checkpoint / Undo
Persist agent-induced changes as discrete checkpoints (commits, snapshots) so any step can be rolled back.
- **Examples**: Aider's auto-commit per edit, Cursor checkpoints, Codex shadow git, Claude Code's "new commit, never amend" guidance.
- **Why it matters**: Recovery without manual `git reflog` archaeology; encourages bolder experimentation.

### 28. User-in-the-Loop Questions
Structured asks back to the user mid-task, with constrained choices to keep responses parseable.
- **Examples**: `AskUserQuestion`, BMAD elicitation prompts, Cursor's clarification modals, java-spring-ai-ask-user-tool.
- **Why it matters**: Bridges ambiguity without dumping a wall of text on the user.

### 29. Output Routing / Context-Saving Sandboxes
Heavy tool output is processed in a side channel; only the agent's summary enters the main context.
- **Examples**: `context-mode` execute/execute_file, `Explore` sub-agent reading excerpts, MCP servers that paginate.
- **Why it matters**: Keeps the main loop fast and cheap even when tools produce megabytes.

### 30. Permission / Policy Cascade
Settings flow from system → org → user → project → run, with explicit precedence rules.
- **Examples**: Claude Code `settings.json` layering, Cursor team policies, Aider config files, Codex profiles.
- **Why it matters**: Centralized governance without blocking individual flexibility.

### 31. Telemetry / Evaluation Harness
A built-in or adjacent system to score agent runs — correctness, cost, latency, drift.
- **Examples**: autobench, prompt-score, agent-intent-eval, agent-semantic-similarity-eval, agent-memory-bench, local-agent-orama.
- **Why it matters**: You cannot improve a harness you cannot measure; eval loops drive iteration.

### 32. Identity / Role Personas
Named personas with embedded expertise, voice, and toolset that the user can summon.
- **Examples**: SuperClaude personas, BMAD roles (PM/Architect/Dev/QA), `*-developer-agent` skills, Claude Code sub-agent definitions.
- **Why it matters**: Mental shorthand for the user; consistent behavior across sessions.

## Cross-Cutting Themes

- **Determinism at the edges, flexibility in the middle**: hooks, configs, and permission gates are deterministic; model reasoning is not. Push policy outward.
- **Externalize state**: memory files, task lists, plan documents, commits — anything the model needs across compactions or sessions.
- **Composability over monoliths**: skills, MCP servers, sub-agents, and hooks all plug into a small core.
- **Blast-radius awareness**: every pattern above either reduces, contains, or makes reversible the side effects of an autonomous loop.

## See Also

- [agents-pocs.md](./agents-pocs.md) — full POC catalog backing these patterns
- [continuous-claude.md](./continuous-claude.md) — background/loop pattern in depth
- [claude-managed-agents.md](./claude-managed-agents.md) — sandbox/worktree isolation
- [SDD-research.md](./SDD-research.md) — spec-driven flows
- [frontend-tools-models-coding-agents.md](./frontend-tools-models-coding-agents.md) — harness comparison
