#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== checkout into a temp folder =="
CHECKOUT=$("$HERE/skill/scripts/checkout.sh" HEAD | grep "^CHECKOUT " | cut -d' ' -f2)
if [ ! -d "$CHECKOUT" ]; then
  echo "FAIL checkout did not produce a folder"
  exit 1
fi
case "$CHECKOUT" in
  "$HERE"*) echo "FAIL checkout landed inside the repo: $CHECKOUT"; exit 1 ;;
esac
echo "OK source checked out to $CHECKOUT"

if "$HERE/skill/scripts/checkout.sh" no-such-branch-xyz >/dev/null 2>&1; then
  echo "FAIL checkout accepted a branch that does not exist"
  exit 1
fi
echo "OK checkout rejects an unknown branch"

echo ""
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
OUT=$(node "$HERE/skill/scripts/render.mjs" "$HERE/sample/triage.json" | grep "^REPORT " | cut -d' ' -f2)
OUT="$(dirname "$OUT")"
if [ ! -f "$OUT/index.html" ]; then
  echo "FAIL report not written"
  exit 1
fi
case "$OUT" in
  "$HERE"*) echo "FAIL report landed inside the repo: $OUT"; exit 1 ;;
  "${TMPDIR%/}"*|/tmp/*|/private/var/folders/*) ;;
  *) echo "FAIL report is not in a temp folder: $OUT"; exit 1 ;;
esac
echo "OK report written to a temp folder"

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

"$HERE/skill/scripts/checkout.sh" --clean >/dev/null

echo "PASS report written to $OUT/index.html"
