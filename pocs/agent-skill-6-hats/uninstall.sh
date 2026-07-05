#!/usr/bin/env bash
set -eu

remove_from() {
  target="$1"
  if [ -e "$target" ]; then
    rm -rf "$target"
    printf 'Removed %s\n' "$target"
  else
    printf 'Not installed at %s\n' "$target"
  fi
}

printf 'Uninstall Hats globally from:\n1) Claude Code\n2) Codex\n3) Both\nChoice: '
read -r choice

case "$choice" in
  1) remove_from "$HOME/.claude/skills/hats" ;;
  2) remove_from "${CODEX_HOME:-$HOME/.codex}/skills/hats" ;;
  3)
    remove_from "$HOME/.claude/skills/hats"
    remove_from "${CODEX_HOME:-$HOME/.codex}/skills/hats"
    ;;
  *) printf 'Invalid choice\n' >&2; exit 1 ;;
esac
