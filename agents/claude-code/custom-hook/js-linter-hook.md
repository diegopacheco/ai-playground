# JS Linter Hook

Runs ESLint on JavaScript files whenever they are edited.

## Setup

1. Create the hook script at `~/.claude/hooks/eslint-hook.sh`:

```bash
#!/bin/bash
input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

if [[ -n "$file_path" && "$file_path" == *.js ]]; then
  npx eslint "$file_path"
fi
```

2. Make it executable:

```bash
chmod +x ~/.claude/hooks/eslint-hook.sh
```

3. Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {
            "type": "command",
            "command": "/Users/YOUR_USERNAME/.claude/hooks/eslint-hook.sh"
          }
        ]
      }
    ]
  }
}
```
