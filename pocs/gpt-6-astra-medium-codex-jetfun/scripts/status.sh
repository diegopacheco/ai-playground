#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
pid="$(port_pid)"
if [ -n "$pid" ]; then printf 'WEB %s UP pid=%s\n' "$WEB" "$pid"; else printf 'WEB %s DOWN pid=-\n' "$WEB"; fi
