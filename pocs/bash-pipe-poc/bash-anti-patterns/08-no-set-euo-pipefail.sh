#!/usr/bin/env bash
# Anti-pattern: omitting 'set -euo pipefail' at the top of the script
# Source: design-doc.md §2.3 and §2.4, countered by G8
# Why it breaks: a failing command (mvn returns 1) is silently passed
# over; the next line keeps going; the script exits 0 and the orchestrator
# records 'tests pass' even though tests never ran.

set -euo pipefail

echo "ANTI-PATTERN: missing 'set -euo pipefail'"
echo "WHY: failures are swallowed silently; 'tests pass' becomes a lie."
echo
echo "DEMO: same logic, with and without strict mode."

echo
echo "Without strict mode:"
bash -c '
build() { return 1; }
build
echo "  build returned non-zero but we kept going"
echo "  exiting 0 -- orchestrator will believe everything is fine"
'
echo "  outer exit code: $?"

echo
echo "With strict mode (set -e):"
set +e
bash -c '
set -euo pipefail
build() { return 1; }
build
echo "  THIS LINE NEVER PRINTS"
'
rc=$?
set -e
echo "  outer exit code: $rc -- the failure is now LOUD"
