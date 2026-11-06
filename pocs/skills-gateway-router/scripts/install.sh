#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
agents="$(agents_for "${1:-all}")"
for agent in $agents; do
  target="$(agent_skills_dir "$agent")/$SKILL_NAME"
  if [ -e "$target" ] && ! is_our_install "$target"; then
    fail "$target exists and was not installed from $ROOT, remove it first"
  fi
  mkdir -p "$target"
  sed "s|\${CLAUDE_SKILL_DIR}/../../../gateway.js|\"$ROOT/gateway.js\"|" "$SKILL_SRC" >"$target/SKILL.md"
  is_our_install "$target" || fail "install to $target did not produce a valid SKILL.md"
  log "$agent: installed $SKILL_NAME into $target"
done
