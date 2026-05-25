#!/usr/bin/env bash
set -u

skill_name="terminal-rpg-agent"
source_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
codex_root="$HOME/.codex/skills"
claude_root="$HOME/.claude/skills"
target_choice=""

choose_target() {
  printf 'Install for:\n'
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

confirm_replace() {
  target="$1"
  if [ -d "$target" ]; then
    printf '%s exists. Replace it? y/N: ' "$target"
    read -r answer
    case "$answer" in
      y|Y|yes|YES) return 0 ;;
      *) return 1 ;;
    esac
  fi
  return 0
}

copy_skill() {
  root="$1"
  target="$root/$skill_name"
  if ! confirm_replace "$target"; then
    printf 'Skipped %s\n' "$target"
    return
  fi
  tmp="$target.tmp.$$"
  rm -rf "$tmp"
  mkdir -p "$tmp/scripts" "$tmp/agents" "$tmp/web"
  cp "$source_dir/SKILL.md" "$tmp/SKILL.md"
  cp "$source_dir/README.md" "$tmp/README.md"
  cp "$source_dir/design-doc.md" "$tmp/design-doc.md"
  cp "$source_dir/scripts/rpg_learn.sh" "$tmp/scripts/rpg_learn.sh"
  cp "$source_dir/agents/openai.yaml" "$tmp/agents/openai.yaml"
  cp "$source_dir/web/index.html" "$tmp/web/index.html"
  cp "$source_dir/web/styles.css" "$tmp/web/styles.css"
  cp "$source_dir/web/game.js" "$tmp/web/game.js"
  chmod +x "$tmp/scripts/rpg_learn.sh"
  rm -rf "$target"
  mkdir -p "$root"
  mv "$tmp" "$target"
  printf 'Installed %s\n' "$target"
}

choose_target
case "$target_choice" in
  codex) copy_skill "$codex_root" ;;
  claude) copy_skill "$claude_root" ;;
  both)
    copy_skill "$codex_root"
    copy_skill "$claude_root"
    ;;
esac
