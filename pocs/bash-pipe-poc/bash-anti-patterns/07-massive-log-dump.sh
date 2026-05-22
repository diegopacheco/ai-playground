#!/usr/bin/env bash
# Anti-pattern: returning the full build/test log to the model
# Source: design-doc.md §2.3, countered by G9 and G14
# Why it breaks: a long Maven/sbt log can be thousands of lines. Feeding
# all of it back blows the context budget and buries the actual error
# line under noise. Tail-on-failure preserves the signal.

set -euo pipefail

echo "ANTI-PATTERN: dumping full build/test logs to the model"
echo "WHY: thousands of lines of noise drown the one line that matters."
echo
echo "DEMO: synthesize a realistic 5000-line build log with the real error"
echo "near the end. Compare what 'cat' vs 'tail -n 20' returns."

log=$(mktemp)
for i in $(seq 1 4998); do echo "[INFO] Downloading $i/5000 ..." ; done > "$log"
echo "[ERROR] Failed to execute goal: compilation failure in Foo.java:42" >> "$log"
echo "[ERROR] BUILD FAILURE" >> "$log"

full_lines=$(wc -l < "$log" | tr -d ' ')
full_bytes=$(wc -c < "$log" | tr -d ' ')
tail_lines=$(tail -n 20 "$log" | wc -l | tr -d ' ')
tail_bytes=$(tail -n 20 "$log" | wc -c | tr -d ' ')

echo
echo "  full dump:       $full_lines lines, $full_bytes bytes"
echo "  tail -n 20:      $tail_lines lines, $tail_bytes bytes"
echo
echo "  tail -n 20 actually shows the error:"
tail -n 3 "$log" | sed 's/^/    /'

rm -f "$log"
echo
echo "Correct form (G14): <=80 lines on failure, <=20 on success."
