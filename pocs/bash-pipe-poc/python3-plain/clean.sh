#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f run.pid ] && bash stop.sh || true
rm -rf .venv __pycache__ .pytest_cache *.log *.pid .build.env .tests.env .start.env .hc.env .stop.env .hc.code .hc.attempts .stop.orphans result.fragment.json
echo "clean ok"
