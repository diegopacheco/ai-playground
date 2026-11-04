---
name: decision-log
description: Record, list, show and supersede architecture decisions that persist across sessions in the shared agent infra state store. Use when the user says "record this decision", "log an ADR", "what did we decide about", "list decisions", or "supersede decision".
---

# Decision Log

Keeps a durable log of architecture decisions. Decisions are stored in the agent infra state
service under this skill's own tenant (`decision-log`), so they survive across sessions and no
other skill or MCP server can read or change them. Every write is also sent to the centralized
logs.

All commands run `scripts/decision.py` from this skill's base directory with Python 3.15 and print JSON.

## Commands

| Goal | Command |
|---|---|
| Record a decision | `python3.15 scripts/decision.py add --title "<title>" --decision "<what>" --why "<why>" [--status proposed\|accepted\|rejected]` |
| Show one decision | `python3.15 scripts/decision.py show <id>` |
| List decisions | `python3.15 scripts/decision.py list [--status accepted]` |
| Supersede a decision | `python3.15 scripts/decision.py supersede <old-id> --by <new-id>` |
| Remove a decision | `python3.15 scripts/decision.py remove <id>` |

The id is the slug of the title: `Use Postgres RLS` becomes `use-postgres-rls`.

## Rules

1. Before recording, run `list` and check that the decision is not already there.
2. Always fill `--why` with the reason the user gave. Ask for it when missing.
3. When a new decision replaces an old one, record the new one first, then `supersede` the old one.
4. If a command fails with `unreachable`, tell the user to run `./scripts/start-all.sh`.
5. Show the user the id, status and title of what changed.
