#!/bin/bash

HOOK_FILE="$HOME/.claude/hooks/permissions.py"
SETTINGS_FILE="$HOME/.claude/settings.json"

if [ -f "$HOOK_FILE" ]; then
    rm "$HOOK_FILE"
    echo "removed $HOOK_FILE"
else
    echo "hook file not found, skipping"
fi

if [ -f "$SETTINGS_FILE" ]; then
    cp "$SETTINGS_FILE" "$SETTINGS_FILE.bak"
    python3 -c "
import json, os

settings_file = os.path.expanduser('$SETTINGS_FILE')
with open(settings_file, 'r') as f:
    settings = json.load(f)

if 'hooks' in settings and 'PreToolUse' in settings['hooks']:
    filtered = []
    for entry in settings['hooks']['PreToolUse']:
        dominated = False
        for h in entry.get('hooks', []):
            if 'permissions.py' in h.get('command', ''):
                dominated = True
                break
        if not dominated:
            filtered.append(entry)
    settings['hooks']['PreToolUse'] = filtered
    if not settings['hooks']['PreToolUse']:
        del settings['hooks']['PreToolUse']
    if not settings['hooks']:
        del settings['hooks']

with open(settings_file, 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')

print('Hook removed from settings.json')
"
    echo "permissions hook uninstalled"
else
    echo "settings.json not found, nothing to clean"
fi
