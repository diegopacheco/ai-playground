#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if port_up "$REPORT"; then
  echo "report  $REPORT  UP    pid $(pid_of report)"
else
  echo "report  $REPORT  DOWN"
fi
for agent in claude codex; do
  if [ -f "$HOME/.$agent/skills/skill-sec-checker/SKILL.md" ]; then
    echo "skill   $agent  INSTALLED"
  else
    echo "skill   $agent  NOT INSTALLED"
  fi
done
