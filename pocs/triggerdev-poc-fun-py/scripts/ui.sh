#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

port_up "$(service_port ui)" || fail "ui is not running, run ./scripts/start-all.sh"
open_url "$(service_url ui)"
