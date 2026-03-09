#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/skill-evaluator"

CLAUDE_TARGET="$HOME/.claude/skills/skill-evaluator"
CODEX_TARGET="$HOME/.codex/skills/skill-evaluator"

mkdir -p "$HOME/.claude/skills"
mkdir -p "$HOME/.codex/skills"

rm -rf "$CLAUDE_TARGET"
cp -r "$SOURCE_DIR" "$CLAUDE_TARGET"
if [ $? -eq 0 ]; then
  echo "[OK] Installed to Claude Code: $CLAUDE_TARGET"
else
  echo "[FAIL] Could not install to Claude Code"
fi

rm -rf "$CODEX_TARGET"
cp -r "$SOURCE_DIR" "$CODEX_TARGET"
if [ $? -eq 0 ]; then
  echo "[OK] Installed to Fox Codex: $CODEX_TARGET"
else
  echo "[FAIL] Could not install to Fox Codex"
fi
