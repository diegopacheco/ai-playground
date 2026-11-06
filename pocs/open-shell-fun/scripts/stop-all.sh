#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_bg frontend
stop_bg backend
log "OpenShell gateway is a shared brew service and keeps running, stop it with: brew services stop openshell"
