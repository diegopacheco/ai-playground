#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port_up "$(service_port ui)" || fail "ui is down, run ./scripts/start-all.sh first"
open_url "$(service_url ui /)"
