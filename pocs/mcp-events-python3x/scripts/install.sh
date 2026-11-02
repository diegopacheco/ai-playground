#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

install_claude() {
  require claude
  claude mcp remove --scope user "$MCP_NAME" >/dev/null 2>&1 || true
  claude mcp add --transport http --scope user "$MCP_NAME" "$(mcp_url)" || fail "claude mcp add failed"
  log "claude code: $MCP_NAME -> $(mcp_url)"
}

install_codex() {
  require codex
  codex mcp remove "$MCP_NAME" >/dev/null 2>&1 || true
  codex mcp add "$MCP_NAME" --url "$(mcp_url)" || fail "codex mcp add failed"
  log "codex: $MCP_NAME -> $(mcp_url)"
}

targets="$(pick_targets Install "$@")"
[ -n "$targets" ] || { log "nothing selected"; exit 0; }
for target in $targets; do
  case "$target" in
    claude) install_claude ;;
    codex) install_codex ;;
    *) fail "unknown target $target, use claude or codex" ;;
  esac
done
port_up "$(service_port backend)" || log "backend is down, run ./scripts/start-all.sh before using the MCP"
