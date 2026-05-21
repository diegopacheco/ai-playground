#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$HERE/results.json"
PROBLEMS="$HERE/problems.md"

EXPECT_STACKS=8
EXPECT_MAX_WALL_CLOCK_SEC=900
STACK_NAMES="python3-plain django-python3 java8-mvn-sb2 java25-mvn-sb4 java25-kotlin-mvn-sb4 java25-scala3-sbt-sb4 scala3-sbt scala2-bazel"

expected_tests_for() {
  case "$1" in
    python3-plain)         echo 3 ;;
    django-python3)        echo 3 ;;
    java8-mvn-sb2)         echo 3 ;;
    java25-mvn-sb4)        echo 3 ;;
    java25-kotlin-mvn-sb4) echo 3 ;;
    java25-scala3-sbt-sb4) echo 3 ;;
    scala3-sbt)            echo 3 ;;
    scala2-bazel)          echo 3 ;;
    *) echo 0 ;;
  esac
}

if [ ! -f "$RESULTS" ]; then
  echo "JUDGE: FAIL"
  echo "  results.json missing at $RESULTS"
  exit 2
fi

FAILS=()

field() {
  local stack="$1" group="$2" key="$3"
  jq -r --arg s "$stack" --arg g "$group" --arg k "$key" '.stacks[$s][$g][$k]' "$RESULTS"
}

for s in $STACK_NAMES; do
  exists=$(jq -r --arg s "$s" '.stacks | has($s)' "$RESULTS")
  if [ "$exists" != "true" ]; then
    FAILS+=("$s.<missing>  expected=present actual=absent")
    continue
  fi
  bp=$(field "$s" build pass)
  br=$(field "$s" build exit_code)
  [ "$bp" = "true" ] || FAILS+=("$s.build.pass  expected=true actual=$bp (exit=$br)")

  tp=$(field "$s" tests pass)
  tt=$(field "$s" tests total)
  tf=$(field "$s" tests failed)
  tsk=$(field "$s" tests skipped)
  exp=$(expected_tests_for "$s")
  [ "$tp" = "true" ] || FAILS+=("$s.tests.pass  expected=true actual=$tp")
  [ "$tt" = "$exp" ] || FAILS+=("$s.tests.total  expected=$exp actual=$tt")
  [ "$tf" = "0" ] || FAILS+=("$s.tests.failed  expected=0 actual=$tf")
  [ "$tsk" = "0" ] || FAILS+=("$s.tests.skipped  expected=0 actual=$tsk")

  sp=$(field "$s" start pass)
  [ "$sp" = "true" ] || FAILS+=("$s.start.pass  expected=true actual=$sp")

  hp=$(field "$s" hc pass)
  hc=$(field "$s" hc status_code)
  [ "$hp" = "true" ] || FAILS+=("$s.hc.pass  expected=true actual=$hp")
  [ "$hc" = "200" ] || FAILS+=("$s.hc.status_code  expected=200 actual=$hc")

  pp=$(field "$s" stop pass)
  porph=$(field "$s" stop orphans)
  [ "$pp" = "true" ] || FAILS+=("$s.stop.pass  expected=true actual=$pp")
  [ "$porph" = "0" ] || FAILS+=("$s.stop.orphans  expected=0 actual=$porph")
done

total_stacks=$(jq -r '.summary.total_stacks' "$RESULTS")
passed_stacks=$(jq -r '.summary.passed_stacks' "$RESULTS")
total_tests=$(jq -r '.summary.total_tests' "$RESULTS")
total_tests_passed=$(jq -r '.summary.total_tests_passed' "$RESULTS")
wall=$(jq -r '.wall_clock_sec' "$RESULTS")

[ "$total_stacks" = "$EXPECT_STACKS" ] || FAILS+=("summary.total_stacks  expected=$EXPECT_STACKS actual=$total_stacks")
[ "$passed_stacks" = "$EXPECT_STACKS" ] || FAILS+=("summary.passed_stacks  expected=$EXPECT_STACKS actual=$passed_stacks")
[ "$wall" -le "$EXPECT_MAX_WALL_CLOCK_SEC" ] || FAILS+=("wall_clock_sec  expected<=$EXPECT_MAX_WALL_CLOCK_SEC actual=$wall")

exp_total=0
for s in $STACK_NAMES; do exp_total=$((exp_total + $(expected_tests_for "$s"))); done
[ "$total_tests" = "$exp_total" ] || FAILS+=("summary.total_tests  expected=$exp_total actual=$total_tests")

if [ "${#FAILS[@]}" -gt 0 ] && [ ! -f "$PROBLEMS" ]; then
  FAILS+=("problems.md  expected=present actual=absent (orchestrator must emit problems.md on any failure)")
fi

if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "JUDGE: PASS — $passed_stacks/$EXPECT_STACKS stacks, $total_tests_passed/$exp_total tests, wall=${wall}s (cap=${EXPECT_MAX_WALL_CLOCK_SEC}s)"
  exit 0
fi

echo "JUDGE: FAIL"
for f in "${FAILS[@]}"; do
  echo "  $f"
done
if [ -f "$PROBLEMS" ]; then echo "problems.md present: yes"; else echo "problems.md present: no"; fi
exit 2
