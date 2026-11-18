#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for name in $(service_names); do
  port="$(service_port "$name")"
  case "$name" in
    postgres) if container_running; then state="UP"; pid="container $POSTGRES_CONTAINER"; else state="DOWN"; pid="-"; fi ;;
    *) pid="$(port_pid "$port")"; if [ -n "$pid" ]; then state="UP"; else state="DOWN"; pid="-"; fi ;;
  esac
  printf "%-16s %-6s %-5s %s\n" "$name" "$port" "$state" "$pid"
done
