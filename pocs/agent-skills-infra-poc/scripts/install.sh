#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

action="${1:-install}"
SKILLS="decision-log answer-grader"

case "$action" in
  install)
    mkdir -p "$ROOT/.claude/skills"
    for skill in $SKILLS; do
      ln -sfn "../../skills/$skill" "$ROOT/.claude/skills/$skill"
      log "skill $skill linked into .claude/skills"
    done
    cat >"$ROOT/.mcp.json" <<JSON
{
  "mcpServers": {
    "bookmarks": { "command": "bash", "args": ["mcp/bookmarks-mcp/run.sh"] },
    "feature-flags": { "command": "bash", "args": ["mcp/feature-flags-mcp/run.sh"] }
  }
}
JSON
    log "mcp servers bookmarks and feature-flags written to .mcp.json"
    log "open Claude Code in this folder to use them"
    ;;
  uninstall)
    for skill in $SKILLS; do
      rm -f "$ROOT/.claude/skills/$skill"
    done
    rm -f "$ROOT/.mcp.json"
    log "skills and mcp servers removed"
    ;;
  *)
    fail "usage: scripts/install.sh [install|uninstall]"
    ;;
esac
