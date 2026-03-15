#!/bin/bash

HOOK_FILE="$HOME/.claude/hooks/permissions.py"
SETTINGS_FILE="$HOME/.claude/settings.json"
CONTEXT_MODE_HOOK="$HOME/.claude/plugins/cache/claude-context-mode/context-mode/0.7.2/hooks/pretooluse.sh"
CONTEXT_MODE_HOOK_MKT="$HOME/.claude/plugins/marketplaces/claude-context-mode/hooks/pretooluse.sh"

POST_HOOK_FILE="$HOME/.claude/hooks/permissions_post.py"
APPROVED_FILE="$HOME/.claude/hooks/permissions_approved.json"

for f in "$HOOK_FILE" "$POST_HOOK_FILE" "$APPROVED_FILE"; do
    if [ -f "$f" ]; then
        rm "$f"
        echo "removed $f"
    fi
done

if [ -f "$SETTINGS_FILE" ]; then
    cp "$SETTINGS_FILE" "$SETTINGS_FILE.bak"
    python3 -c "
import json, os

settings_file = os.path.expanduser('$SETTINGS_FILE')
with open(settings_file, 'r') as f:
    settings = json.load(f)

for hook_type in ['PreToolUse', 'PostToolUse']:
    if 'hooks' in settings and hook_type in settings['hooks']:
        filtered = []
        for entry in settings['hooks'][hook_type]:
            keep = True
            for h in entry.get('hooks', []):
                if 'permissions' in h.get('command', ''):
                    keep = False
                    break
            if keep:
                filtered.append(entry)
        settings['hooks'][hook_type] = filtered
        if not settings['hooks'][hook_type]:
            del settings['hooks'][hook_type]
if 'hooks' in settings and not settings['hooks']:
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

restore_context_mode() {
    local hookfile="$1"
    if [ -f "$hookfile.bak" ]; then
        cp "$hookfile.bak" "$hookfile"
        rm "$hookfile.bak"
        echo "Restored context-mode: $hookfile"
    fi
}

restore_context_mode "$CONTEXT_MODE_HOOK"
restore_context_mode "$CONTEXT_MODE_HOOK_MKT"

AVFILE="$HOME/.claude/hooks/approve-variants.py"
if [ -f "$AVFILE.bak" ]; then
    cp "$AVFILE.bak" "$AVFILE"
    rm "$AVFILE.bak"
    echo "Restored approve-variants.py"
fi
