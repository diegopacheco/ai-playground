#!/usr/bin/env bash
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir="$root/hats"

install_to() {
  target="$1"
  if [ -e "$target" ]; then
    printf '%s already exists. Replace it? [y/N] ' "$target"
    read -r replace
    case "$replace" in
      y|Y) rm -rf "$target" ;;
      *) printf 'Skipped %s\n' "$target"; return ;;
    esac
  fi
  mkdir -p "$(dirname -- "$target")"
  cp -R "$source_dir" "$target"
  printf 'Installed %s\n' "$target"
}

printf 'Install Hats globally for:\n1) Claude Code\n2) Codex\n3) Both\nChoice: '
read -r choice

case "$choice" in
  1) install_to "$HOME/.claude/skills/hats" ;;
  2) install_to "${CODEX_HOME:-$HOME/.codex}/skills/hats" ;;
  3)
    install_to "$HOME/.claude/skills/hats"
    install_to "${CODEX_HOME:-$HOME/.codex}/skills/hats"
    ;;
  *) printf 'Invalid choice\n' >&2; exit 1 ;;
esac
