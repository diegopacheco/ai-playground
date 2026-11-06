#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for name in $(service_names); do
  port="$(service_port "$name")"
  pid="$(port_pid "$port")"
  if [ -n "$pid" ]; then
    printf "%-10s %-6s %-5s pid %s\n" "$name" "$port" UP "$pid"
  else
    printf "%-10s %-6s %-5s\n" "$name" "$port" DOWN
  fi
done
