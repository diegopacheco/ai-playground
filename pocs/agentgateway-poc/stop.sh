#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

pkill -f "port-forward .*agentgateway-proxy" 2>/dev/null || true
pkill -f "node .*strip-proxy.js" 2>/dev/null || true
ccr stop >/dev/null 2>&1 || true
rm -f /tmp/agentgateway-pf.pid /tmp/agentgateway-ui-pf.pid /tmp/strip-proxy.pid

if kind get clusters | grep -qx agentgateway; then
  kind delete cluster --name agentgateway
fi

echo "agentgateway cluster removed"
