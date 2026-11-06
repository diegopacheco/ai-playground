#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require node
node gateway.js sources
for agent in claude codex; do
  target="$(agent_skills_dir "$agent")/$SKILL_NAME"
  if is_our_install "$target"; then log "$agent INSTALLED $target"; else log "$agent NOT INSTALLED"; fi
done
