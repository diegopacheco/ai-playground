#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
bp_use_java "$JDK_ID"
LOG="$HERE/test.log"
t0=$(bp_now_sec)
RC=0
eval "$TEST_CMD" >"$LOG" 2>&1 || RC=$?
t1=$(bp_now_sec)
dur=$((t1 - t0))
eval "$(bp_parse_scalatest "$LOG")"
PASS=false
[ "$RC" -eq 0 ] && [ "$TOTAL" -gt 0 ] && [ "$FAILED" -eq 0 ] && PASS=true
{
  echo "TESTS_PASS=$PASS"
  echo "TESTS_TOTAL=$TOTAL"
  echo "TESTS_PASSED=$PASSED"
  echo "TESTS_FAILED=$FAILED"
  echo "TESTS_SKIPPED=$SKIPPED"
  echo "TESTS_DURATION=$dur"
} > .tests.env
[ "$RC" -ne 0 ] && bp_log_tail "$LOG" 80
exit "$RC"
