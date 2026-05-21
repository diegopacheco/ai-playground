#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
bp_use_java "$JDK_ID"
sbt $SBT_OPTS_BP update compile
CP=$(sbt $SBT_OPTS_BP "export Runtime / fullClasspath" 2>/dev/null | tail -n 1)
echo "$CP" > runtime-classpath.txt
echo "setup ok cp_len=$(echo -n "$CP" | wc -c | tr -d ' ')"
