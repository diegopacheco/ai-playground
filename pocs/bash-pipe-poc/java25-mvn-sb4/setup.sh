#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source ./bash-pipe.env
source ../lib/common.sh
bp_use_java "$JDK_ID"
mvn -B -ntp -q dependency:go-offline -DexcludeArtifactIds=tomcat-embed-core || true
echo "setup ok JAVA_HOME=$JAVA_HOME"
