# Direct Skill 🎯

A skill for Claude Code and Codex that makes answers direct: no metaphors, no filler,
2-5 lines, full paths for every file mentioned.

## Trigger

You call it, it never turns itself on. It stays on until you say stop.

- explain in simple terms
- tell me in simple terms
- explain simply
- explain simple
- tell me how this works
- use direct skill / use reply skill / be direct
- `/direct` or `$direct`

Nothing is written to `~/.claude/CLAUDE.md` or `~/.codex/AGENTS.md`.

## Rules it enforces

- First line answers the question.
- 2-5 lines. More only if asked.
- No metaphors, no analogies, no over-intellectualizing.
- Still correct: real technical terms, one short clause explaining each.
- Files always as full absolute paths.
- No closing summary, no "hope this helps".

It changes every word said to you, including reports of what changed. Code and tests stay complete.

## Install

```bash
./install.sh
```

Choose `1` Claude Code, `2` Codex, or `3` both.

- Claude Code: `~/.claude/skills/direct`
- Codex: `~/.codex/skills/direct` (or `$CODEX_HOME/skills/direct`)

## Uninstall

```bash
./uninstall.sh
```

Same 1 / 2 / 3 choice.

## Layout

```
/Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agent-skill-direct
├── direct/SKILL.md
├── install.sh
├── uninstall.sh
└── README.md
```
