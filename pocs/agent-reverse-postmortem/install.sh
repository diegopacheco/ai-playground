#!/bin/bash
set -e

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_ROOT="$HOME/.claude/skills"
SKILLS="reverse-post reverse-post-site"

for SKILL in $SKILLS; do
  SKILL_DIR="$SKILLS_ROOT/$SKILL"
  if [ -d "$SKILL_DIR" ]; then
    echo "Updating existing skill '$SKILL' at $SKILL_DIR"
    rm -rf "$SKILL_DIR"
  fi
  mkdir -p "$SKILL_DIR"
  cp "$SOURCE_DIR/$SKILL/SKILL.md" "$SKILL_DIR/SKILL.md"
  echo "Installed '$SKILL' to $SKILL_DIR"
done

echo ""
echo "Done. Use /reverse-post to analyze, then /reverse-post-site to render the report."
