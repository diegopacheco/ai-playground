#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skill_name="codebase-roast-map"

write_entry() {
  dir="$1"
  script="$2"
  mode="$3"
  name="$4"
  mkdir -p "$dir"
  printf '%s\n' "Use the codebase-roast-map skill. Run this command from the repository root:" "" "node \"$script\" $mode \"\$PWD\"" > "$dir/$name.md"
}

install_codex() {
  skill_dir="$HOME/.agents/skills/$skill_name"
  rm -rf "$skill_dir"
  mkdir -p "$HOME/.agents/skills"
  cp -R "$root/skill" "$skill_dir"
  chmod +x "$skill_dir/scripts/roast-map.mjs"
  rm -rf "$HOME/.codex/skills/$skill_name"
  mkdir -p "$HOME/.codex/skills"
  cp -R "$root/skill" "$HOME/.codex/skills/$skill_name"
  chmod +x "$HOME/.codex/skills/$skill_name/scripts/roast-map.mjs"
  rm -f "$HOME/.codex/prompts/roast.md"
  rm -f "$HOME/.codex/prompts/roast-map.md"
  printf '%s\n' "Installed for Codex"
  printf '%s\n' "Use /skills and choose codebase-roast-map"
  printf '%s\n' "Or type: \$codebase-roast-map roast this repo"
  printf '%s\n' "Codex CLI does not support custom slash commands"
}

install_claude() {
  skill_dir="$HOME/.claude/skills/$skill_name"
  rm -rf "$skill_dir"
  mkdir -p "$HOME/.claude/skills"
  cp -R "$root/skill" "$skill_dir"
  chmod +x "$skill_dir/scripts/roast-map.mjs"
  write_entry "$HOME/.claude/commands" "$skill_dir/scripts/roast-map.mjs" "report" "roast"
  write_entry "$HOME/.claude/commands" "$skill_dir/scripts/roast-map.mjs" "map" "roast-map"
  printf '%s\n' "Installed for Claude Code"
  printf '%s\n' "Commands: /roast and /roast-map"
  printf '%s\n' "Restart Claude Code if it was already open"
}

printf '%s\n' "Install codebase-roast-map for?"
printf '%s\n' "1) Codex"
printf '%s\n' "2) Claude Code"
printf '%s\n' "3) Both"
printf '%s' "> "
read -r choice

case "$choice" in
  1) install_codex ;;
  2) install_claude ;;
  3) install_codex; install_claude ;;
  *) printf '%s\n' "Invalid choice"; exit 1 ;;
esac
