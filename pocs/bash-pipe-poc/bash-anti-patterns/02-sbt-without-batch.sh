#!/usr/bin/env bash
# Anti-pattern: invoking sbt without --batch
# Source: design-doc.md §2.2 + §2.3, countered by G1 and G13
# Why it breaks: bare 'sbt' opens an interactive REPL and waits on stdin
# forever. The Bash tool cannot type 'exit' into it, so the call hangs
# until the outer 10-minute ceiling expires.

set -euo pipefail

echo "ANTI-PATTERN: sbt without --batch"
echo "WHY: bare sbt opens a REPL and blocks on stdin; the Bash tool will hang."
echo
echo "DEMO: simulating a tool that blocks on stdin (no --batch)."
echo "We wrap it in a 1s timeout so this script does not actually hang."

t0=$(date +%s)
fake_sbt_repl() { read -r _line; }
if timeout 1 bash -c 'fake_sbt_repl() { read -r _line; }; fake_sbt_repl'; then
  echo "RESULT: returned cleanly (unexpected)"
else
  rc=$?
  t1=$(date +%s)
  echo "RESULT: killed by timeout after $((t1 - t0))s, exit=$rc"
  echo "Real sbt would block here until the Bash tool ceiling (~10 min) fires."
fi
