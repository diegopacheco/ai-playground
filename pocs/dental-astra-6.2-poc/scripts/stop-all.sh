#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if running; then kill "$(cat .run/web.pid)"; fi
rm -f .run/web.pid
