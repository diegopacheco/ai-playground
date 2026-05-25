#!/usr/bin/env bash
set -u

skill_name="terminal-rpg-agent"
codex_target="$HOME/.codex/skills/$skill_name"
claude_target="$HOME/.claude/skills/$skill_name"
state_root="$HOME/.terminal-rpg-agent"
target_choice=""

choose_target() {
  printf 'Uninstall from:\n'
  printf '1. Codex\n'
  printf '2. Claude\n'
  printf '3. Both\n'
  printf 'Pick 1-3: '
  read -r choice
  case "$choice" in
    1) target_choice="codex" ;;
    2) target_choice="claude" ;;
    3) target_choice="both" ;;
    *) printf 'Invalid choice\n' >&2; exit 1 ;;
  esac
}

remove_target() {
  target="$1"
  if [ ! -d "$target" ]; then
    printf 'Not installed: %s\n' "$target"
    return
  fi
  printf 'Remove %s? y/N: ' "$target"
  read -r answer
  case "$answer" in
    y|Y|yes|YES)
      rm -rf "$target"
      printf 'Removed %s\n' "$target"
      ;;
    *)
      printf 'Skipped %s\n' "$target"
      ;;
  esac
}

choose_target
case "$target_choice" in
  codex) remove_target "$codex_target" ;;
  claude) remove_target "$claude_target" ;;
  both)
    remove_target "$codex_target"
    remove_target "$claude_target"
    ;;
esac

if [ -d "$state_root" ]; then
  printf 'Remove run history in %s? y/N: ' "$state_root"
  read -r state_answer
  case "$state_answer" in
    y|Y|yes|YES)
      rm -rf "$state_root"
      printf 'Removed %s\n' "$state_root"
      ;;
    *)
      printf 'Kept %s\n' "$state_root"
      ;;
  esac
fi
