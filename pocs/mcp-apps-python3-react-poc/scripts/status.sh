#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

printf "%-14s %-8s %-6s %s\n" SERVICE PORT STATE PID
port="$(service_port calculator)"
pid="$(port_pid "$port")"
printf "%-14s %-8s %-6s %s\n" calculator "$port" "$([ -n "$pid" ] && echo UP || echo DOWN)" "${pid:--}"
printf "%-14s %-8s %-6s %s\n" funnel "$(service_port funnel)" "$(funnel_up && echo UP || echo DOWN)" "-"
