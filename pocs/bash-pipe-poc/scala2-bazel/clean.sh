#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[ -f run.pid ] && bash stop.sh || true
bazel shutdown >/dev/null 2>&1 || true
bazel clean --expunge >/dev/null 2>&1 || true
rm -rf bazel-* *.log *.pid .build.env .tests.env .start.env .hc.env .stop.env .hc.code .hc.attempts .stop.orphans result.fragment.json
echo "clean ok"
