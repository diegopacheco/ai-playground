#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f run.pid ] && bash stop.sh || true
rm -rf target *.log *.pid .build.env .tests.env .start.env .hc.env .stop.env .hc.code .hc.attempts .stop.orphans result.fragment.json
echo "clean ok"
