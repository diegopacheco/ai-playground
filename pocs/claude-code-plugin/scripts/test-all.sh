#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
require_claude

failures=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1" >&2; failures=$((failures + 1)); }

if claude plugin validate . >/dev/null 2>&1; then
  pass "marketplace manifest is valid"
else
  fail "marketplace manifest is valid"
fi

for p in $PLUGINS; do
  if grep -q "\"source\": \"./plugins/$p\"" .claude-plugin/marketplace.json; then
    pass "$p is listed in the marketplace so it can be picked on its own"
  else
    fail "$p is listed in the marketplace so it can be picked on its own"
  fi

  if claude plugin validate "plugins/$p" >/dev/null 2>&1; then
    pass "$p plugin manifest is valid"
  else
    fail "$p plugin manifest is valid"
  fi

  if grep -q "\"name\": \"$p\"" "plugins/$p/.claude-plugin/plugin.json"; then
    pass "$p plugin.json name matches the marketplace entry"
  else
    fail "$p plugin.json name matches the marketplace entry"
  fi

  count="$(find "plugins/$p/skills" -mindepth 2 -maxdepth 2 -name SKILL.md | wc -l | tr -d ' ')"
  if [ "$count" -eq 3 ]; then
    pass "$p ships exactly 3 skills"
  else
    fail "$p ships exactly 3 skills (found $count)"
  fi

  for skill in "plugins/$p"/skills/*/SKILL.md; do
    dir="$(basename "$(dirname "$skill")")"
    if grep -q "^name: $dir$" "$skill" && grep -q "^description: " "$skill"; then
      pass "$p:$dir has a name matching its folder and a description for auto selection"
    else
      fail "$p:$dir has a name matching its folder and a description for auto selection"
    fi
  done
done

if [ "$failures" -gt 0 ]; then
  echo "$failures test(s) failed" >&2
  exit 1
fi
echo "all tests passed"
