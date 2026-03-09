#!/bin/bash
CLAUDE_TARGET="$HOME/.claude/skills/skill-evaluator"
CODEX_TARGET="$HOME/.codex/skills/skill-evaluator"

if [ -d "$CLAUDE_TARGET" ]; then
  rm -rf "$CLAUDE_TARGET"
  echo "[OK] Removed from Claude Code: $CLAUDE_TARGET"
else
  echo "[SKIP] Not installed in Claude Code"
fi

if [ -d "$CODEX_TARGET" ]; then
  rm -rf "$CODEX_TARGET"
  echo "[OK] Removed from Fox Codex: $CODEX_TARGET"
else
  echo "[SKIP] Not installed in Fox Codex"
fi
