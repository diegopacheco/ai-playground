#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require tailscale

case "${1:-on}" in
  on)
    host="$(public_host)"
    if [ -z "$host" ]; then
      host="$(tailscale status --json | grep -m1 '"DNSName"' | sed -E 's/.*"DNSName": *"([^"]*)\.".*/\1/')"
      [ -n "$host" ] || fail "could not read the MagicDNS name from tailscale status"
      printf "MCP_PUBLIC_HOST=%s\n" "$host" >"$SCRIPTS/tailscale.env"
      log "wrote scripts/tailscale.env"
    fi
    tailscale funnel --bg --https="$(service_port funnel)" "$(service_port calculator)" >/dev/null || fail "tailscale funnel failed, approve Funnel in the Tailscale admin console"
    "$SCRIPTS/stop-all.sh"
    "$SCRIPTS/start-all.sh" >/dev/null
    log "funnel on, add this URL to ChatGPT: $(public_url "$host")"
    ;;
  off)
    tailscale funnel --https="$(service_port funnel)" off >/dev/null 2>&1 || true
    log "funnel off"
    ;;
  *)
    fail "usage: funnel.sh on|off"
    ;;
esac
