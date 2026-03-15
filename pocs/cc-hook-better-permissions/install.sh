#!/bin/bash

HOOK_DIR="$HOME/.claude/hooks"
SETTINGS_FILE="$HOME/.claude/settings.json"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT_MODE_HOOK="$HOME/.claude/plugins/cache/claude-context-mode/context-mode/0.7.2/hooks/pretooluse.sh"
CONTEXT_MODE_HOOK_MKT="$HOME/.claude/plugins/marketplaces/claude-context-mode/hooks/pretooluse.sh"

mkdir -p "$HOOK_DIR"

cp "$SCRIPT_DIR/permissions.py" "$HOOK_DIR/permissions.py"
chmod +x "$HOOK_DIR/permissions.py"

if [ -f "$SETTINGS_FILE" ]; then
    cp "$SETTINGS_FILE" "$SETTINGS_FILE.bak"
fi

python3 -c "
import json, os, re

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

patch_approve_variants() {
    local avfile="$HOME/.claude/hooks/approve-variants.py"
    if [ -f "$avfile" ]; then
        if grep -q 'curl' "$avfile"; then
            sed -i.bak 's/|curl//g' "$avfile"
            echo "Patched approve-variants.py: removed curl from safe list"
        fi
    fi
}

patch_context_mode() {
    local hookfile="$1"
    if [ ! -f "$hookfile" ]; then
        return
    fi
    cp "$hookfile" "$hookfile.bak"
    python3 -c "
import re

with open('$hookfile', 'r') as f:
    content = f.read()

content = re.sub(
    r'  # curl/wget.*?exit 0\n  fi\n',
    '',
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'# ─── WebFetch: deny.*?exit 0\nfi',
    '# ─── WebFetch: passthrough ───\nif [ \"\\\$TOOL\" = \"WebFetch\" ]; then\n  exit 0\nfi',
    content,
    flags=re.DOTALL
)

with open('$hookfile', 'w') as f:
    f.write(content)
"
    echo "Patched context-mode: $hookfile"
    echo "  backup: $hookfile.bak"
}

patch_approve_variants

if [ -f "$CONTEXT_MODE_HOOK" ]; then
    patch_context_mode "$CONTEXT_MODE_HOOK"
fi
if [ -f "$CONTEXT_MODE_HOOK_MKT" ]; then
    patch_context_mode "$CONTEXT_MODE_HOOK_MKT"
fi

echo ""
echo "permissions hook installed"
echo "  hook: $HOOK_DIR/permissions.py"
echo "  config: $SETTINGS_FILE"
if [ -f "$SETTINGS_FILE.bak" ]; then
    echo "  backup: $SETTINGS_FILE.bak"
fi
