#!/bin/bash

SKILL_DIR="$HOME/.claude/skills/threat-analyst"

if [ -d "$SKILL_DIR" ]; then
    rm -rf "$SKILL_DIR"
    echo "threat-analyst skill removed from $SKILL_DIR"
else
    echo "threat-analyst skill not found at $SKILL_DIR"
fi
