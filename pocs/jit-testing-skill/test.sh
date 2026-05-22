#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JIT="${SCRIPT_DIR}/bin/jit"

fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "ok: $*"; }

run_snapshot_sample() {
  local sample="$1"
  local expected="$2"
  local needs="$3"
  if [ -n "${needs}" ] && ! command -v "${needs}" >/dev/null 2>&1; then
    echo "skip ${sample}: ${needs} not installed"
    return
  fi
  local work
  work="$(mktemp -d)"
  cp -R "${SCRIPT_DIR}/samples/${sample}/." "${work}/"
  cd "${work}"
  local detected
  detected="$("${JIT}" detect .)"
  [ "${detected}" = "${expected}" ] || fail "${sample}: expected ${expected}, got ${detected}"
  pass "${sample}: detected ${expected}"
  "${JIT}" run --repo . --workflow dodgy >/tmp/jit-run.out
  grep -q "mode: snapshot" /tmp/jit-run.out || fail "${sample}: not using snapshot mode"
  grep -q "catches:" /tmp/jit-run.out || fail "${sample}: no run summary"
  pass "${sample}: snapshot pipeline ran"
  cd - >/dev/null
  rm -rf "${work}"
}

run_git_sample() {
  command -v git >/dev/null 2>&1 || { echo "skip git sample: git not installed"; return; }
  local work
  work="$(mktemp -d)"
  cp -R "${SCRIPT_DIR}/samples/python-pricing/." "${work}/"
  cd "${work}"
  ./setup-git.sh >/dev/null
  rm -rf .parent
  "${JIT}" run --repo . --diff HEAD~1..HEAD --workflow dodgy --mode git >/tmp/jit-run.out
  grep -q "mode: git" /tmp/jit-run.out || fail "git fallback not used"
  pass "git fallback still works"
  cd - >/dev/null
  rm -rf "${work}"
}

run_snapshot_sample "python-pricing" "python3" "python3"
run_snapshot_sample "nodejs-discount" "nodejs" "node"
run_snapshot_sample "java-shipping" "java8" "javac"
run_snapshot_sample "scala-discount" "scala3-sbt" ""
run_snapshot_sample "kotlin-discount" "kotlin" ""

run_git_sample

echo ""
echo "all smoke checks passed"
