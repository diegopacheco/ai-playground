---
name: skills-gateway
description: Gateway and router to a catalog of skills that are not registered directly. Use it before any task a specialized skill could handle, such as text, JSON or git chores. It finds the right skill and loads its instructions.
allowed-tools: Bash(node:*)
---

# Skills Gateway

The skills behind this gateway are not registered with you. This gateway is the only way to reach them.

Every command runs as:

```bash
node ${CLAUDE_SKILL_DIR}/../../../gateway.js <command>
```

1. Find candidates with `route "<the request in a few words>"`. It prints the best matching skills, best first, named `<source>__<skill>`.
2. If no candidate fits, run `list` to see the whole catalog. If nothing fits there either, tell the user no skill covers it and do the task without the gateway.
3. Load the chosen skill with `load <source>__<skill> <arguments>`. It prints the skill instructions and its base directory.
4. Follow the loaded instructions exactly as if that skill had been invoked directly. Load only the skills the task needs.
5. If a command exits non-zero, tell the user the stderr message. Exit 2 is an unknown skill or a bad request, exit 3 is an unreachable skill source.
