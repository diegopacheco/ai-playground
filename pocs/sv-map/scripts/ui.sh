#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port_up "$(service_port WEB)" || fail "WEB is down, run scripts/start-all.sh"
open_url "$(service_url WEB)"
