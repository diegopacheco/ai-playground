#!/usr/bin/env bash
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir="$root/direct"

install_to() {
  target="$1"
  if [ -e "$target" ]; then
    printf '⚠️  %s already exists. Replace it? [y/N] ' "$target"
    read -r replace
    case "$replace" in
      y|Y) rm -rf "$target" ;;
      *) printf '⏭️  Skipped %s\n' "$target"; return ;;
    esac
  fi
  mkdir -p "$(dirname -- "$target")"
  cp -R "$source_dir" "$target"
  printf '✅ Installed %s\n' "$target"
}

printf '🎯 Direct Skill — straight answers, no metaphors, 2-5 lines\n\n'
printf 'Install globally for:\n  1) 🤖 Claude Code\n  2) 🧠 Codex\n  3) 🔥 Both\nChoice: '
read -r choice

case "$choice" in
  1) install_to "$HOME/.claude/skills/direct" ;;
  2) install_to "${CODEX_HOME:-$HOME/.codex}/skills/direct" ;;
  3)
    install_to "$HOME/.claude/skills/direct"
    install_to "${CODEX_HOME:-$HOME/.codex}/skills/direct"
    ;;
  *) printf '❌ Invalid choice\n' >&2; exit 1 ;;
esac
