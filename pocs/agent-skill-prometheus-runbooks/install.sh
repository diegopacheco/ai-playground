#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/skill/prometheus-runbooks"
DEST="$HOME/.claude/skills/prometheus-runbooks"
mkdir -p "$HOME/.claude/skills"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"
echo "Installed prometheus-runbooks skill to $DEST"
echo "Files:"
find "$DEST" -type f
