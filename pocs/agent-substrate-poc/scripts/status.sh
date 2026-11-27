#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

printf "%-14s %-8s %-6s %s\n" SERVICE PORT STATE PID
for name in $(service_names); do
  port="$(service_port "$name")"
  pid="$(port_pid "$port")"
  if [ -n "$pid" ]; then
    printf "%-14s %-8s %-6s %s\n" "$name" "$port" UP "$pid"
  else
    printf "%-14s %-8s %-6s %s\n" "$name" "$port" DOWN "-"
  fi
done
if kube -n ate-system get deploy ate-api-server >/dev/null 2>&1; then
  printf "%-14s %-8s %-6s %s\n" substrate "-" UP "$KCTX"
else
  printf "%-14s %-8s %-6s %s\n" substrate "-" DOWN "$KCTX"
fi
