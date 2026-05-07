#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TRAILBLAZE_BIN="$HOME/.trailblaze/bin/trailblaze"

if [ ! -x "$TRAILBLAZE_BIN" ]; then
  echo "trailblaze not installed. Run ./install.sh first."
  exit 1
fi

./start.sh

set +e
"$TRAILBLAZE_BIN" trail \
  --device web \
  --driver PLAYWRIGHT_NATIVE \
  --use-recorded-steps \
  --llm ollama/gpt-oss:20b \
  --headless \
  --no-report \
  --no-daemon \
  trails/web.trail.yaml
RESULT=$?
set -e

./stop.sh

if [ "$RESULT" -eq 0 ]; then
  echo "TRAIL PASSED"
else
  echo "TRAIL FAILED (exit $RESULT)"
fi
exit "$RESULT"
