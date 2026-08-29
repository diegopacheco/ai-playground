#!/usr/bin/env bash
set -eu

remove_from() {
  target="$1"
  if [ -e "$target" ]; then
    rm -rf "$target"
    printf '🗑️  Removed %s\n' "$target"
  else
    printf 'ℹ️  Not installed at %s\n' "$target"
  fi
}

printf '🎯 Direct Skill — uninstall\n\n'
printf 'Remove globally from:\n  1) 🤖 Claude Code\n  2) 🧠 Codex\n  3) 🔥 Both\nChoice: '
read -r choice

case "$choice" in
  1) remove_from "$HOME/.claude/skills/direct" ;;
  2) remove_from "${CODEX_HOME:-$HOME/.codex}/skills/direct" ;;
  3)
    remove_from "$HOME/.claude/skills/direct"
    remove_from "${CODEX_HOME:-$HOME/.codex}/skills/direct"
    ;;
  *) printf '❌ Invalid choice\n' >&2; exit 1 ;;
esac
