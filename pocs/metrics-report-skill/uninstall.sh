#!/bin/bash
SKILL_DIR="$HOME/.claude/skills/metrics-report"

if [ -d "$SKILL_DIR" ]; then
  rm -rf "$SKILL_DIR"
  echo "metrics-report skill uninstalled from $SKILL_DIR"
else
  echo "metrics-report skill is not installed"
fi
