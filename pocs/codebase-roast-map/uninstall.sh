#!/usr/bin/env bash
set -euo pipefail

skill_name="codebase-roast-map"

uninstall_codex() {
  rm -rf "$HOME/.agents/skills/$skill_name"
  rm -rf "$HOME/.codex/skills/$skill_name"
  rm -f "$HOME/.codex/prompts/roast.md"
  rm -f "$HOME/.codex/prompts/roast-map.md"
  printf '%s\n' "Removed from Codex"
}

uninstall_claude() {
  rm -rf "$HOME/.claude/skills/$skill_name"
  rm -f "$HOME/.claude/commands/roast.md"
  rm -f "$HOME/.claude/commands/roast-map.md"
  printf '%s\n' "Removed from Claude Code"
}

printf '%s\n' "Uninstall codebase-roast-map from?"
printf '%s\n' "1) Codex"
printf '%s\n' "2) Claude Code"
printf '%s\n' "3) Both"
printf '%s' "> "
read -r choice

case "$choice" in
  1) uninstall_codex ;;
  2) uninstall_claude ;;
  3) uninstall_codex; uninstall_claude ;;
  *) printf '%s\n' "Invalid choice"; exit 1 ;;
esac
