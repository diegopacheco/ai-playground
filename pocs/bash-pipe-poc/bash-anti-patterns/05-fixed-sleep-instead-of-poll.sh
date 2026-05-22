#!/usr/bin/env bash
# Anti-pattern: 'sleep N && curl' instead of a bounded poll loop
# Source: design-doc.md §2.3, countered by G4
# Why it breaks: a fixed sleep is either too short (flaky on slow boots)
# or too long (wastes wall clock when the server is up in 200ms).

set -euo pipefail

echo "ANTI-PATTERN: fixed 'sleep N && curl' for readiness"
echo "WHY: too short => flaky; too long => wasted wall clock. Either way, wrong."
echo
echo "DEMO: file-based readiness signal so the demo is portable and fast."

ready_file=$(mktemp -u /tmp/bp_ap05_ready.XXXXXX)

echo
echo "Case A: server takes 1s; we wait 0s -- too short, FLAKY"
rm -f "$ready_file"
( sleep 1; touch "$ready_file" ) &
pid=$!
t0=$(date +%s%N)
[ -f "$ready_file" ] && echo "  ready" || echo "  NOT ready -- the server was still starting"
t1=$(date +%s%N)
echo "  elapsed_ms=$(( (t1 - t0) / 1000000 ))"
wait "$pid" 2>/dev/null || true
rm -f "$ready_file"

echo
echo "Case B: server is up instantly; we still 'sleep 1' -- WASTED time"
rm -f "$ready_file"
( touch "$ready_file" ) &
pid=$!
t0=$(date +%s%N)
sleep 1
[ -f "$ready_file" ] && echo "  ready" || echo "  NOT ready"
t1=$(date +%s%N)
echo "  elapsed_ms=$(( (t1 - t0) / 1000000 )) -- the server was ready almost immediately"
wait "$pid" 2>/dev/null || true
rm -f "$ready_file"

echo
echo "Case C (correct, G4): bounded poll loop -- returns as soon as ready"
rm -f "$ready_file"
( sleep 1; touch "$ready_file" ) &
pid=$!
t0=$(date +%s%N)
for i in $(seq 1 5); do
  if [ -f "$ready_file" ]; then
    t1=$(date +%s%N); echo "  ready after $i tries, elapsed_ms=$(( (t1 - t0) / 1000000 ))"; break
  fi
  sleep 1
done
wait "$pid" 2>/dev/null || true
rm -f "$ready_file"
