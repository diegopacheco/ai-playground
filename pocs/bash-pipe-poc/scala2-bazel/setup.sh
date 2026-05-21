#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
bp_use_java "$JDK_ID"
bazel fetch $BAZEL_OPTS_BP //:app //:app_tests
echo "setup ok"
