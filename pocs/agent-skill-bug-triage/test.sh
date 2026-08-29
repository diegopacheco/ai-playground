#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/bug-triage-report"

echo "== existing suite (green while the bug is live) =="
node --test "$HERE/sample/test/cart.test.mjs" 2>&1 | grep -E "^# (pass|fail)|^. (pass|fail)|^ℹ (pass|fail)"
SUITE=$?

echo ""
echo "== reproduction test (must fail) =="
node --test "$HERE/sample/test/repro-negative-total.test.mjs" >/tmp/bug-triage-repro.log 2>&1
REPRO=$?
grep -E "^ℹ (pass|fail)" /tmp/bug-triage-repro.log
if [ "$REPRO" -eq 0 ]; then
  echo "FAIL reproduction test passed, the bug is not reproduced"
  exit 1
fi
echo "OK reproduction test fails as expected"

echo ""
echo "== render report =="
rm -rf "$OUT"
node "$HERE/skill/scripts/render.mjs" "$HERE/sample/triage.json" "$OUT"
if [ ! -f "$OUT/index.html" ]; then
  echo "FAIL report not written"
  exit 1
fi

for anchor in id=\"name\" id=\"description\" id=\"files\" id=\"repro\" id=\"why\" id=\"solution\" id=\"touch\" id=\"breaking\" id=\"safety\"; do
  if ! grep -q "$anchor" "$OUT/index.html"; then
    echo "FAIL missing section $anchor"
    exit 1
  fi
done

if ! grep -q "atlassian.net/browse/PIX-482" "$OUT/index.html"; then
  echo "FAIL tracker link missing"
  exit 1
fi

echo "PASS report written to $OUT/index.html"
