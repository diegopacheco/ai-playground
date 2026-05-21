#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
t0=$(bp_now_sec)
RC=0
bp_wait_hc "http://127.0.0.1:$PORT$HC_PATH" "$HERE/run.log" 60 || RC=$?
t1=$(bp_now_sec)
PASS=false
[ "$RC" -eq 0 ] && PASS=true
{
  echo "HC_PASS=$PASS"
  echo "HC_CODE=$(cat .hc.code 2>/dev/null || echo 0)"
  echo "HC_ATTEMPTS=$(cat .hc.attempts 2>/dev/null || echo 0)"
  echo "HC_DURATION=$((t1 - t0))"
} > .hc.env
exit "$RC"
