#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
LOG="$HERE/test.log"
t0=$(bp_now_sec)
RC=0
eval "$TEST_CMD" >"$LOG" 2>&1 || RC=$?
t1=$(bp_now_sec)
dur=$((t1 - t0))
RAN=$( { grep -oE 'Ran [0-9]+ tests' "$LOG" || true; } | grep -oE '[0-9]+' | head -1 || true)
RAN=${RAN:-0}
STATUS=$(grep -E '^(OK|FAILED)' "$LOG" | tail -n 1 || true)
FAILED=0
if echo "$STATUS" | grep -q '^FAILED'; then
  FAILED=$( { echo "$STATUS" | grep -oE 'failures=[0-9]+' || true; } | grep -oE '[0-9]+' | head -1 || true)
  FAILED=${FAILED:-1}
fi
PASSED=$((RAN - FAILED))
PASS=false
[ "$RC" -eq 0 ] && [ "$RAN" -gt 0 ] && [ "$FAILED" -eq 0 ] && PASS=true
{
  echo "TESTS_PASS=$PASS"
  echo "TESTS_TOTAL=$RAN"
  echo "TESTS_PASSED=$PASSED"
  echo "TESTS_FAILED=$FAILED"
  echo "TESTS_SKIPPED=0"
  echo "TESTS_DURATION=$dur"
} > .tests.env
[ "$RC" -ne 0 ] && bp_log_tail "$LOG" 60
exit "$RC"
