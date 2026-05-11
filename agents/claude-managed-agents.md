# Claude Managed Agents

Announced: 08.APR.2026 (public beta)

Source: https://claude.com/blog/claude-managed-agents
Docs: https://platform.claude.com/docs/en/managed-agents/overview

## What it is

A suite of composable APIs for building and deploying cloud-hosted agents on Anthropic's infrastructure. You define the agent's tasks, tools, and guardrails. Anthropic runs the orchestration harness (decides when to call tools, manages context, recovers from errors) on their infra.

Positioned as an alternative to the typical DIY agent stack where you rebuild sandboxing, checkpointing, credentials, scoped permissions, and tracing every time, plus rework the loop on every model upgrade.

## Capabilities

* Production runtime: sandboxed code execution, authentication, tool execution.
* Long-running sessions: hours of autonomous work, persists through disconnections.
* Multi-agent coordination: agents spin up and direct sub-agents (research preview).
* Governance: scoped permissions, identity management, execution tracing.
* Outcome-driven loop: define success criteria, Claude self-evaluates and iterates (research preview). Also supports prompt-and-response.
* Console observability: session traces, integration analytics, troubleshooting, every tool call and decision inspectable.

Internal benchmark claim: up to +10 points on structured file generation vs a standard prompting loop, biggest gains on the hardest problems.

## Pricing

* Standard Claude Platform token rates, plus
* $0.08 per session-hour of active runtime.

## How to start

* Claude Console -> agent-quickstart workspace
* New CLI to deploy the first agent
* Claude Code with the claude-api Skill: prompt "start onboarding for managed agents in Claude API"

## Who is using it

* Notion: delegate work inside workspaces (private alpha in Notion Custom Agents). Dozens of tasks in parallel.
* Rakuten: enterprise agents across product, sales, marketing, finance into Slack and Teams. Each specialist agent deployed within a week.
* Asana: AI Teammates working alongside humans inside Asana projects.
* Vibecode: Managed Agents as the default integration, prompt to deployed app.
* Atlassian: developer agents embedded in Jira so users assign tasks from Jira.

## Relation to existing Anthropic offerings

* Sits on top of the Claude API. The model API is unchanged. Managed Agents adds harness, runtime, and ops plane around it.
* Complements Claude Code. Claude Code with the claude-api Skill is the recommended on-ramp for building a Managed Agent.
* vs OpenAI AgentKit (see open-ai-agent-kit.md): AgentKit is a visual drag and drop with guardrails, MCP, and state. Managed Agents is API/CLI/code-first with cloud runtime, long sessions, sub-agent spawning, and per session-hour billing. Aimed at developers shipping production systems, not drag and drop prototyping.

## Notes

* The $0.08/session-hour is on top of token cost. Long-running idle agents accrue runtime cost even when not heavily inferring. Cost model differs from raw token billing.
* Multi-agent coordination and outcome-driven loops are gated behind a research preview request form: http://claude.com/form/claude-managed-agents
* Public beta means breaking changes are possible. Worth pinning versions when used in real workloads.

## PROS

* Removes the infra rebuild on every agent project.
* Long-running sessions survive disconnects, useful for hours-long jobs.
* Built-in tracing in the Console means no DIY OpenTelemetry plumbing to see tool calls and failures.
* Native fit with Claude Code for onboarding.
* Sub-agent spawning is a first-class feature, not bolted on.

## CONS

* Lock-in to Anthropic infra. Hard to lift and shift to another provider.
* Per session-hour billing adds a second cost axis to reason about beyond tokens.
* Multi-agent and outcome-driven mode are behind a gated request form, not generally available.
* Public beta, so surface area can change.
* No self-hosting path documented.
