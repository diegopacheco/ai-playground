#!/usr/bin/env bash
set -euo pipefail

NS=agentgateway-system

pkill -f "port-forward .*agentgateway-proxy.*15000" 2>/dev/null || true

kubectl port-forward -n "$NS" deployment/agentgateway-proxy 15000 &
PID=$!
echo $PID > /tmp/agentgateway-ui-pf.pid

until curl -sf -o /dev/null -m 1 http://localhost:15000/ui/ || [ $? -eq 22 ]; do sleep 1; done

echo "agentgateway admin UI: http://localhost:15000/ui/"
echo "press Ctrl+C to stop"
wait $PID
