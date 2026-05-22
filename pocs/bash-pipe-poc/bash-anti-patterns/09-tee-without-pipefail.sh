#!/usr/bin/env bash
# Anti-pattern: 'cmd | tee log' without 'set -o pipefail'
# Source: design-doc.md §2.4, countered by G8
# Why it breaks: the pipeline's exit code is tee's (always 0), not cmd's.
# A failing mvn|tee build.log reports success.

set -euo pipefail

echo "ANTI-PATTERN: piping to tee without pipefail masks the real exit code"
echo "WHY: pipeline exit is the LAST command's exit; tee is always 0."
echo
echo "DEMO: 'false | tee /dev/null' under both modes."

echo
echo "Without pipefail:"
set +o pipefail
set +e
false | tee /dev/null >/dev/null
echo "  exit code: $? (zero -- the false was hidden by tee)"
set -e

echo
echo "With pipefail:"
set -o pipefail
set +e
false | tee /dev/null >/dev/null
echo "  exit code: $? (non-zero -- failure surfaced)"
set -e
