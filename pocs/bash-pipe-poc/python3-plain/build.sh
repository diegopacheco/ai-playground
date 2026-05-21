#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
LOG="$HERE/build.log"
t0=$(bp_now_sec)
RC=0
eval "$BUILD_CMD" >"$LOG" 2>&1 || RC=$?
t1=$(bp_now_sec)
dur=$((t1 - t0))
if [ "$RC" -eq 0 ]; then
  echo "BUILD_PASS=true" > .build.env
else
  echo "BUILD_PASS=false" > .build.env
  bp_log_tail "$LOG" 40
fi
echo "BUILD_DURATION=$dur" >> .build.env
echo "BUILD_RC=$RC" >> .build.env
exit "$RC"
