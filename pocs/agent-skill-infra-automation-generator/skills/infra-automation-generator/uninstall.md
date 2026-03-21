# Uninstall infra-automation-generator

## Steps

1. Remove the skill directory:
```bash
rm -rf ~/.claude/skills/infra-automation-generator
```

2. If you added the skill to a project-level `.claude/settings.json`, remove the `infra-automation-generator` entry from the skills section.

3. Verify removal — restart Claude Code and confirm the skill no longer appears in the available skills list.
