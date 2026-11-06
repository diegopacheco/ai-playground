#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/.run"
LOGS="$RUN/logs"
cd "$ROOT"
mkdir -p "$LOGS"

require() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required but not installed"
}

log() {
  printf "%s\n" "$*"
}

fail() {
  printf "ERROR: %s\n" "$*" >&2
  exit 1
}

SKILL_NAME="skills-gateway"
SKILL_SRC="$ROOT/.claude/skills/$SKILL_NAME/SKILL.md"

agent_skills_dir() {
  case "$1" in
    claude) printf "%s\n" "${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}" ;;
    codex) printf "%s\n" "${CODEX_SKILLS_DIR:-$HOME/.codex/skills}" ;;
    *) fail "unknown agent: $1, use claude, codex or all" ;;
  esac
}

agents_for() {
  case "${1:-all}" in
    all) printf "claude\ncodex\n" ;;
    claude|codex) printf "%s\n" "$1" ;;
    *) fail "unknown agent: $1, use claude, codex or all" ;;
  esac
}

is_our_install() {
  [ -f "$1/SKILL.md" ] && grep -q "^name: $SKILL_NAME$" "$1/SKILL.md" && grep -qF "$ROOT/gateway.js" "$1/SKILL.md"
}
