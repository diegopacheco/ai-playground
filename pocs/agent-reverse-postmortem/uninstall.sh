#!/bin/bash
set -e

SKILLS_ROOT="$HOME/.claude/skills"
SKILLS="reverse-post reverse-post-site"

for SKILL in $SKILLS; do
  SKILL_DIR="$SKILLS_ROOT/$SKILL"
  if [ -d "$SKILL_DIR" ]; then
    rm -rf "$SKILL_DIR"
    echo "Uninstalled '$SKILL' from $SKILL_DIR"
  else
    echo "Skill '$SKILL' is not installed"
  fi
done
