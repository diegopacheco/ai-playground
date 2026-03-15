#!/bin/bash

HOOK_DIR="$HOME/.claude/hooks"
SETTINGS_FILE="$HOME/.claude/settings.json"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOOK_DIR"

cp "$SCRIPT_DIR/permissions.py" "$HOOK_DIR/permissions.py"
chmod +x "$HOOK_DIR/permissions.py"

if [ -f "$SETTINGS_FILE" ]; then
    cp "$SETTINGS_FILE" "$SETTINGS_FILE.bak"
fi

python3 -c "
import json, os, sys

settings_file = os.path.expanduser('$SETTINGS_FILE')
hook_entry = {
    'matcher': '.*',
    'hooks': [
        {
            'type': 'command',
            'command': 'python3 ~/.claude/hooks/permissions.py'
        }
    ]
}

settings = {}
if os.path.exists(settings_file):
    with open(settings_file, 'r') as f:
        settings = json.load(f)

if 'hooks' not in settings:
    settings['hooks'] = {}
if 'PreToolUse' not in settings['hooks']:
    settings['hooks']['PreToolUse'] = []

already_installed = False
for entry in settings['hooks']['PreToolUse']:
    for h in entry.get('hooks', []):
        if 'permissions.py' in h.get('command', ''):
            already_installed = True
            break

if not already_installed:
    settings['hooks']['PreToolUse'].append(hook_entry)

with open(settings_file, 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')

print('Hook registered in settings.json')
"

echo "permissions hook installed"
echo "  hook: $HOOK_DIR/permissions.py"
echo "  config: $SETTINGS_FILE"
if [ -f "$SETTINGS_FILE.bak" ]; then
    echo "  backup: $SETTINGS_FILE.bak"
fi
