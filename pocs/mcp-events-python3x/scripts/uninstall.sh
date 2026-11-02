#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

targets="$(pick_targets Uninstall "$@")"
[ -n "$targets" ] || { log "nothing selected"; exit 0; }
for target in $targets; do
  case "$target" in
    claude)
      require claude
      claude mcp remove --scope user "$MCP_NAME" >/dev/null 2>&1 || true
      log "claude code: $MCP_NAME removed"
      ;;
    codex)
      require codex
      codex mcp remove "$MCP_NAME" >/dev/null 2>&1 || true
      log "codex: $MCP_NAME removed"
      ;;
    *) fail "unknown target $target, use claude or codex" ;;
  esac
done
