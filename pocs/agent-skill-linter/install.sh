#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/skill"
DEST="$HOME/.claude/skills/agent-skill-linter"
CMD="$HOME/.claude/commands"

rm -rf "$DEST"
mkdir -p "$DEST" "$CMD"
cp -R "$SRC/." "$DEST/"
find "$DEST" -type d \( -name node_modules -o -name target -o -name dist \) -prune -exec rm -rf {} +
find "$DEST" -type f \( -name '*.log' -o -name '*.pid' \) -delete
cp "$DEST/commands/lint.md" "$CMD/lint.md"
cp "$DEST/commands/lint-site.md" "$CMD/lint-site.md"

if ! command -v podman > /dev/null 2>&1; then
  echo "warning: podman not found; /lint works but /lint-site needs podman"
fi

echo "installed agent-skill-linter to $DEST"
echo "commands available: /lint  /lint-site"
