#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

agents="$(agents_for "${1:-all}")"
for agent in $agents; do
  target="$(agent_skills_dir "$agent")/$SKILL_NAME"
  if [ ! -e "$target" ]; then
    log "$agent: $SKILL_NAME not installed"
    continue
  fi
  is_our_install "$target" || fail "$target was not installed from $ROOT, refusing to remove it"
  rm -rf "$target"
  log "$agent: removed $target"
done
