#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if owned && healthy; then printf 'web port %s UP pid %s\n' "$WEB" "$(cat "$RUN/web.pid")"; else printf 'web port %s DOWN\n' "$WEB"; fi
