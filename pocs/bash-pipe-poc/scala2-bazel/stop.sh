#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ../lib/common.sh
t0=$(bp_now_sec)
bp_stop_app "$HERE/run.pid"
bazel shutdown >/dev/null 2>&1 || true
t1=$(bp_now_sec)
{
  echo "STOP_PASS=true"
  echo "STOP_DURATION=$((t1 - t0))"
  echo "STOP_ORPHANS=$(cat .stop.orphans 2>/dev/null || echo 0)"
} > .stop.env
