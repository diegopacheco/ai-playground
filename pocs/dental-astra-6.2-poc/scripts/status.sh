#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if running; then echo "Web $WEB_PORT UP pid $(cat .run/web.pid)"; else echo "Web $WEB_PORT DOWN"; fi
