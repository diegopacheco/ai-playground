#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for name in $(service_names); do
  port="$(service_port "$name")"
  if port_up "$port"; then
    log "$name $port UP pid $(port_pid "$port")"
  else
    log "$name $port DOWN"
  fi
done
