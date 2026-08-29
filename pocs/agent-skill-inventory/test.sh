#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCAN="$HERE/skills/inventory/scripts/scan.py"
RENDER="$HERE/skills/inventory/scripts/render.py"
WORK="$(cd "$(mktemp -d "${TMPDIR:-/tmp}/inventory-tests-XXXXXX")" && pwd -P)"
REPO="$WORK/fixture"
OUT="$WORK/report"

PASS=0
FAIL=0

ok()   { PASS=$((PASS+1)); echo "  ✅ $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  ❌ $1"; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected '$3', got '$2')"; fi; }

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "🧪 inventory skill tests"
echo ""
echo "workspace: $WORK"
echo ""

bash "$HERE/tests/fixture.sh" "$REPO" > /dev/null

echo "1. scanner collects facts from the code, not from guesses"
python3 "$SCAN" "$REPO" "$WORK/facts.json" > /dev/null || bad "scan.py exited non-zero"
q() { python3 -c "import json,sys;f=json.load(open('$WORK/facts.json'));print($1)"; }
check "facts.json is valid JSON with a repo root" "$(q "1 if f['repoRoot'] else 0")" "1"
check "finds both migration tables" "$(q "f['schema']['tableCount']")" "2"
check "names the orders table" "$(q "1 if any(t['name']=='orders' for t in f['schema']['tables']) else 0")" "1"
check "counts the 2 rust integration test cases" "$(q "f['tests']['integration']['cases']")" "2"
check "classifies the playwright spec as e2e" "$(q "f['tests']['e2e']['files']")" "1"
check "classifies the k6 script as stress" "$(q "f['tests']['stress']['files']")" "1"
check "does not invent a unit suite" "$(q "'unit' in f['tests']")" "False"
check "counts the TODO marker in the readme" "$(q "f['techDebtSignals']['markerCounts'].get('TODO',0)")" "1"
check "counts the tracing error call" "$(q "f['observability']['logCounts'].get('error',0)")" "1"
check "flags the unbounded orders query" "$(q "1 if any('unbounded' in x['flags'] for x in f['schema']['suspiciousQueries']) else 0")" "1"
check "does not flag the primary key lookup" "$(q "1 if all('WHERE id = \$1' not in x['query'] or x['flags']!=['unbounded'] for x in f['schema']['suspiciousQueries']) else 0")" "1"
check "reads the single fixture commit" "$(q "f['totals']['commits']")" "1"
check "attributes the commit to the fixture author" "$(q "f['committers'][0]['name']")" "Fixture Author"
echo ""

echo "2. renderer builds a report from a valid analysis"
python3 "$HERE/tests/analysis.py" "$WORK/analysis.json" valid
python3 "$RENDER" "$WORK/facts.json" "$WORK/analysis.json" "$OUT" > /dev/null
check "index.html was written" "$([ -f "$OUT/index.html" ] && echo yes || echo no)" "yes"
h() { grep -c "$1" "$OUT/index.html" | tr -d ' '; }
check "the report is one self contained file" "$(ls "$OUT" | grep -vc analysis.json)" "1"
check "the diagram is inline svg" "$([ "$(h '<svg viewBox')" -ge 1 ] && echo yes || echo no)" "yes"
check "the hand drawn wobble filter is present" "$([ "$(h 'feTurbulence')" -ge 1 ] && echo yes || echo no)" "yes"
check "the report directory is written into the page" "$([ "$(h "$OUT")" -ge 1 ] && echo yes || echo no)" "yes"
check "the codebase path is written into the page" "$([ "$(h "$REPO")" -ge 1 ] && echo yes || echo no)" "yes"
check "all seven tabs are declared" "$(grep -o 'panel: "p-[a-z]*"' "$OUT/index.html" | wc -l | tr -d ' ')" "7"
check "no template placeholder is left behind" "$(grep -c '__DATA__\|__DIAGRAM__\|__TITLE__' "$OUT/index.html" | tr -d ' ')" "0"
p() { python3 -c "
import json,re,sys
src=open('$OUT/index.html').read()
d=json.loads(re.search(r'<script id=\"payload\" type=\"application/json\">(.*?)</script>', src, re.S).group(1).replace('<\\\\/','</'))
print($1)"; }
check "the payload carries the report directory" "$(p "1 if d['reportDir']=='$OUT' else 0")" "1"
check "the payload carries 5 pros and 5 cons" "$(p "len(d['modules'][0]['pros']),len(d['modules'][0]['cons'])")" "5 5"
check "the payload carries the committer avatar url" "$(p "1 if d['committers'][0]['avatarUrl'].startswith('https://www.gravatar.com/') else 0")" "1"
check "the payload carries the scan totals" "$(p "d['facts']['totals']['commits']")" "1"
echo ""

echo "3. renderer rejects an analysis that does not hold up"
reject() {
  python3 "$HERE/tests/analysis.py" "$WORK/broken.json" "$1"
  local msg
  msg="$(python3 "$RENDER" "$WORK/facts.json" "$WORK/broken.json" "$WORK/broken-out" 2>&1)"
  local code=$?
  if [ "$code" -eq 0 ]; then
    bad "$2 (renderer accepted it)"
  elif echo "$msg" | grep -qi "$3"; then
    ok "$2"
  else
    bad "$2 (rejected, but the message did not mention '$3')"
  fi
}
reject bad-path        "a file path that does not exist"        "does not exist"
reject four-pros       "a module with only 4 pros"              "exactly 5 are required"
reject unknown-edge    "an architecture edge to a missing node" "unknown node"
reject wrong-commits   "a commit count the scan disagrees with" "the scan counted"
reject one-pass        "an analysis that skipped the passes"    "must be 3"
reject no-evidence     "an architecture edge with no evidence"  "no evidence"
reject eleven-debt     "more than 10 tech debt items in a module" "cap is 10"
reject wrong-test-count "a test file count the scan disagrees with" "the scan counted"
reject wrong-case-count "a test case count the scan disagrees with" "cases, the scan counted"
reject bad-severity    "a severity outside high, medium and low" "severity must be"
echo ""

echo "4. the schema tab is dropped when there is no schema"
python3 "$HERE/tests/analysis.py" "$WORK/noschema.json" no-schema
python3 "$RENDER" "$WORK/facts.json" "$WORK/noschema.json" "$WORK/noschema-out" > /dev/null
n() { python3 -c "
import json,re
src=open('$WORK/noschema-out/index.html').read()
d=json.loads(re.search(r'<script id=\"payload\" type=\"application/json\">(.*?)</script>', src, re.S).group(1).replace('<\\\\/','</'))
print($1)"; }
check "the payload marks the schema absent" "$(n "d['schema']['present']")" "False"
check "the report still renders" "$([ -f "$WORK/noschema-out/index.html" ] && echo yes || echo no)" "yes"
echo ""

echo "5. install and uninstall scripts are valid shell"
for f in install.sh uninstall.sh test.sh tests/fixture.sh; do
  if bash -n "$HERE/$f" 2>/dev/null; then ok "$f parses"; else bad "$f has a syntax error"; fi
done
for f in skills/inventory/scripts/scan.py skills/inventory/scripts/render.py tests/analysis.py; do
  if python3 -c "import ast,sys; ast.parse(open('$HERE/$f').read())" 2>/dev/null; then
    ok "$f parses"
  else
    bad "$f has a syntax error"
  fi
done
echo ""

echo "─────────────────────────────────────────"
echo "passed: $PASS   failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "❌ FAILED"
  exit 1
fi
echo "✅ ALL TESTS PASSED"
