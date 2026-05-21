#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
t0=$(bp_now_sec)
bp_start_app "$RUN_CMD" "$HERE/run.log" "$HERE/run.pid"
t1=$(bp_now_sec)
{
  echo "START_PASS=true"
  echo "START_DURATION=$((t1 - t0))"
  echo "START_PID=$(cat "$HERE/run.pid")"
} > .start.env
