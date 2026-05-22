#!/usr/bin/env bash
# Anti-pattern: no 'trap' installed -> no cleanup on script abort
# Source: design-doc.md §2.4, countered by G8
# Why it breaks: when 'set -e' fires mid-script, temp files, started
# servers, and background jobs leak. The next invocation finds stale
# PID files and port collisions.

set -euo pipefail

echo "ANTI-PATTERN: no EXIT trap to clean up temp state"
echo "WHY: an error mid-script leaves temp files, sockets, and child jobs behind."
echo
echo "DEMO: a script that creates a temp file and then errors out."

echo
echo "Without trap:"
tmp_no_trap=$(mktemp /tmp/bp_ap11_notrap.XXXXXX)
echo "  created: $tmp_no_trap"
bash -c "set -e; false; rm -f '$tmp_no_trap'" || true
if [ -f "$tmp_no_trap" ]; then
  echo "  RESULT: $tmp_no_trap STILL EXISTS (leaked)"
  rm -f "$tmp_no_trap"
fi

echo
echo "With trap (correct, G8):"
tmp_trap=$(mktemp /tmp/bp_ap11_trap.XXXXXX)
echo "  created: $tmp_trap"
bash -c "set -e; trap 'rm -f \"$tmp_trap\"' EXIT; false" || true
if [ -f "$tmp_trap" ]; then
  echo "  RESULT: $tmp_trap leaked (unexpected)"
  rm -f "$tmp_trap"
else
  echo "  RESULT: $tmp_trap cleaned up by trap"
fi
