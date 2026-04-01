#!/bin/bash
SKILL_DIR="$HOME/.claude/skills/metrics-report"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/SKILL.md"

mkdir -p "$SKILL_DIR/metrics-report"
cp -r "$SCRIPT_DIR/metrics-report/metrics-application" "$SKILL_DIR/metrics-report/"
cp -r "$SCRIPT_DIR/metrics-report/data" "$SKILL_DIR/metrics-report/"
cp "$SCRIPT_DIR/metrics-report/metrics-config.json" "$SKILL_DIR/metrics-report/"
cp "$SCRIPT_DIR/metrics-report/run.sh" "$SKILL_DIR/metrics-report/"
cp "$SCRIPT_DIR/metrics-report/stop.sh" "$SKILL_DIR/metrics-report/"

cd "$SKILL_DIR/metrics-report/metrics-application" && npm install

echo "metrics-report skill installed to $SKILL_DIR"
echo "You can now use /metrics-report in Claude Code"
