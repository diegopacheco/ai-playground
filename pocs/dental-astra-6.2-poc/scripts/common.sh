#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/ports.env
service_url(){ echo "http://127.0.0.1:$WEB_PORT"; }
running(){
  [[ -f .run/web.pid ]] && ps -p "$(cat .run/web.pid)" -o command= | grep -Fq "http.server $WEB_PORT --bind 127.0.0.1 --directory $ROOT"
}
