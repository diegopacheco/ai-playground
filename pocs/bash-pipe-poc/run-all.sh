#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

STACKS=(
  python3-plain
  django-python3
  java8-mvn-sb2
  java25-mvn-sb4
  java25-kotlin-mvn-sb4
  java25-scala3-sbt-sb4
  scala3-sbt
  scala2-bazel
)

RESULTS="$HERE/results.json"
PROBLEMS="$HERE/problems.md"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
T0=$(date +%s)

PROBLEMS_BODY=""
FRAGMENTS=()

emit_problem() {
  local stack="$1"
  local script="$2"
  local rc="$3"
  local dur="$4"
  local log="$5"
  local tail_text=""
  if [ -f "$log" ]; then
    tail_text=$(tail -n 20 "$log" | sed 's/^/  /')
  fi
  PROBLEMS_BODY+="

## $stack
- ${script} exit $rc, duration ${dur}s
- Log tail:
\`\`\`
${tail_text}
\`\`\`"
}

env_get() {
  local f="$1"
  local k="$2"
  local v=""
  [ -f "$f" ] && v=$(grep -E "^${k}=" "$f" | head -n1 | cut -d= -f2- || true)
  echo "${v:-}"
}

run_one() {
  local stack="$1"
  local dir="$HERE/$stack"
  echo ">>> $stack"
  ( cd "$dir" && bash setup.sh >/dev/null 2>&1 || true )

  local build_pass=false build_dur=0 build_rc=0
  local tests_pass=false tests_total=0 tests_passed=0 tests_failed=0 tests_skipped=0 tests_dur=0
  local start_pass=false start_dur=0 start_pid=0
  local hc_pass=false hc_code=0 hc_att=0 hc_dur=0
  local stop_pass=false stop_dur=0 stop_orph=0

  ( cd "$dir" && bash build.sh ) >/dev/null 2>&1 || true
  build_pass=$(env_get "$dir/.build.env" BUILD_PASS); build_pass=${build_pass:-false}
  build_dur=$(env_get "$dir/.build.env" BUILD_DURATION); build_dur=${build_dur:-0}
  build_rc=$(env_get "$dir/.build.env" BUILD_RC); build_rc=${build_rc:-0}
  [ "$build_pass" = "false" ] && emit_problem "$stack" "build.sh" "$build_rc" "$build_dur" "$dir/build.log"

  ( cd "$dir" && bash test.sh ) >/dev/null 2>&1 || true
  tests_pass=$(env_get "$dir/.tests.env" TESTS_PASS); tests_pass=${tests_pass:-false}
  tests_total=$(env_get "$dir/.tests.env" TESTS_TOTAL); tests_total=${tests_total:-0}
  tests_passed=$(env_get "$dir/.tests.env" TESTS_PASSED); tests_passed=${tests_passed:-0}
  tests_failed=$(env_get "$dir/.tests.env" TESTS_FAILED); tests_failed=${tests_failed:-0}
  tests_skipped=$(env_get "$dir/.tests.env" TESTS_SKIPPED); tests_skipped=${tests_skipped:-0}
  tests_dur=$(env_get "$dir/.tests.env" TESTS_DURATION); tests_dur=${tests_dur:-0}
  [ "$tests_pass" = "false" ] && emit_problem "$stack" "test.sh" "1" "$tests_dur" "$dir/test.log"

  ( cd "$dir" && bash start.sh ) >/dev/null 2>&1 || true
  start_pass=$(env_get "$dir/.start.env" START_PASS); start_pass=${start_pass:-false}
  start_dur=$(env_get "$dir/.start.env" START_DURATION); start_dur=${start_dur:-0}
  start_pid=$(env_get "$dir/.start.env" START_PID); start_pid=${start_pid:-0}

  ( cd "$dir" && bash hc.sh ) >/dev/null 2>&1 || true
  hc_pass=$(env_get "$dir/.hc.env" HC_PASS); hc_pass=${hc_pass:-false}
  hc_code=$(env_get "$dir/.hc.env" HC_CODE); hc_code=${hc_code:-0}
  hc_att=$(env_get "$dir/.hc.env" HC_ATTEMPTS); hc_att=${hc_att:-0}
  hc_dur=$(env_get "$dir/.hc.env" HC_DURATION); hc_dur=${hc_dur:-0}
  [ "$hc_pass" = "false" ] && emit_problem "$stack" "hc.sh" "1" "$hc_dur" "$dir/run.log"

  ( cd "$dir" && bash stop.sh ) >/dev/null 2>&1 || true
  stop_pass=$(env_get "$dir/.stop.env" STOP_PASS); stop_pass=${stop_pass:-false}
  stop_dur=$(env_get "$dir/.stop.env" STOP_DURATION); stop_dur=${stop_dur:-0}
  stop_orph=$(env_get "$dir/.stop.env" STOP_ORPHANS); stop_orph=${stop_orph:-0}
  [ "$stop_orph" != "0" ] && emit_problem "$stack" "stop.sh" "0" "$stop_dur" "$dir/run.log"

  local frag="$dir/result.fragment.json"
  cat > "$frag" <<EOF
"$stack": {
  "build":  { "pass": $build_pass, "duration_sec": $build_dur, "exit_code": $build_rc },
  "tests":  { "pass": $tests_pass, "total": $tests_total, "passed": $tests_passed, "failed": $tests_failed, "skipped": $tests_skipped, "duration_sec": $tests_dur },
  "start":  { "pass": $start_pass, "duration_sec": $start_dur, "pid": $start_pid },
  "hc":     { "pass": $hc_pass, "status_code": $hc_code, "attempts": $hc_att, "duration_sec": $hc_dur },
  "stop":   { "pass": $stop_pass, "duration_sec": $stop_dur, "orphans": $stop_orph }
}
EOF
  FRAGMENTS+=("$frag")
  echo "    build=$build_pass tests=$tests_passed/$tests_total hc=$hc_pass orphans=$stop_orph"
}

for s in "${STACKS[@]}"; do
  run_one "$s"
done

T1=$(date +%s)
WALL=$((T1 - T0))

TOTAL=${#STACKS[@]}
PASSED_STACKS=0
TOTAL_TESTS=0
TOTAL_TESTS_PASSED=0

stack_body=""
first=1
for f in "${FRAGMENTS[@]}"; do
  if [ "$first" -eq 1 ]; then first=0; else stack_body+=","$'\n'"    "; fi
  stack_body+="$(cat "$f")"
  bp=$(grep -oE '"build":  \{ "pass": (true|false)' "$f" | grep -oE '(true|false)' | head -n1)
  tp=$(grep -oE '"tests":  \{ "pass": (true|false)' "$f" | grep -oE '(true|false)' | head -n1)
  hp=$(grep -oE '"hc":     \{ "pass": (true|false)' "$f" | grep -oE '(true|false)' | head -n1)
  sp=$(grep -oE '"stop":   \{ "pass": (true|false)' "$f" | grep -oE '(true|false)' | head -n1)
  orph=$(grep -oE '"orphans": [0-9]+' "$f" | grep -oE '[0-9]+' | head -n1)
  tt=$(grep -oE '"total": [0-9]+' "$f" | grep -oE '[0-9]+' | head -n1)
  tpassed=$(grep -oE '"passed": [0-9]+' "$f" | grep -oE '[0-9]+' | head -n1)
  TOTAL_TESTS=$((TOTAL_TESTS + ${tt:-0}))
  TOTAL_TESTS_PASSED=$((TOTAL_TESTS_PASSED + ${tpassed:-0}))
  if [ "$bp" = "true" ] && [ "$tp" = "true" ] && [ "$hp" = "true" ] && [ "$sp" = "true" ] && [ "${orph:-0}" = "0" ]; then
    PASSED_STACKS=$((PASSED_STACKS + 1))
  fi
done

cat > "$RESULTS" <<EOF
{
  "schema_version": 1,
  "started_at": "$STARTED_AT",
  "wall_clock_sec": $WALL,
  "stacks": {
    $stack_body
  },
  "summary": {
    "total_stacks": $TOTAL,
    "passed_stacks": $PASSED_STACKS,
    "failed_stacks": $((TOTAL - PASSED_STACKS)),
    "total_tests": $TOTAL_TESTS,
    "total_tests_passed": $TOTAL_TESTS_PASSED
  }
}
EOF

if [ -z "$PROBLEMS_BODY" ]; then
  cat > "$PROBLEMS" <<EOF
# Problems — $STARTED_AT

(none)
EOF
else
  cat > "$PROBLEMS" <<EOF
# Problems — $STARTED_AT
$PROBLEMS_BODY
EOF
fi

echo
echo "wrote $RESULTS"
echo "wrote $PROBLEMS"
echo "passed_stacks=$PASSED_STACKS/$TOTAL tests=$TOTAL_TESTS_PASSED/$TOTAL_TESTS wall=${WALL}s"
