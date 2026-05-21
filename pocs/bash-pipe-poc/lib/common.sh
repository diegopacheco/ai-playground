set -euo pipefail

bp_log_tail() {
  local f="$1"
  local n="${2:-40}"
  [ -f "$f" ] && tail -n "$n" "$f" || echo "(no log: $f)"
}

bp_start_app() {
  local cmd="$1"
  local logf="$2"
  local pidf="$3"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "already running pid=$(cat "$pidf")"
    return 0
  fi
  : > "$logf"
  nohup bash -c "$cmd" > "$logf" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$pidf"
  echo "started pid=$pid log=$logf"
}

bp_wait_hc() {
  local url="$1"
  local logf="$2"
  local max="${3:-60}"
  local i=0
  local code=000
  local attempts=0
  while [ "$i" -lt "$max" ]; do
    attempts=$((attempts + 1))
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url" || echo "000")
    if [ "$code" = "200" ]; then
      echo "healthy after ${attempts}s code=200"
      echo "$attempts" > .hc.attempts
      echo "200" > .hc.code
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "unhealthy after ${max}s code=$code"
  echo "$attempts" > .hc.attempts
  echo "$code" > .hc.code
  bp_log_tail "$logf" 80
  return 1
}

bp_stop_app() {
  local pidf="$1"
  if [ ! -f "$pidf" ]; then
    echo "not running"
    echo "0" > .stop.orphans
    return 0
  fi
  local pid
  pid=$(cat "$pidf")
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  local i=0
  while [ "$i" -lt 5 ]; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
    i=$((i + 1))
  done
  kill -KILL "$pid" 2>/dev/null || true
  pkill -KILL -P "$pid" 2>/dev/null || true
  sleep 1
  sleep 1
  local orphans
  orphans=$({ pgrep -P "$pid" 2>/dev/null || true; } | wc -l | tr -d ' ')
  echo "$orphans" > .stop.orphans
  rm -f "$pidf"
  echo "stopped pid=$pid orphans=$orphans"
}

bp_emit_fragment() {
  local stack="$1"
  local out="$2"
  local b_pass="${BUILD_PASS:-false}"
  local b_dur="${BUILD_DURATION:-0}"
  local b_rc="${BUILD_RC:-0}"
  local t_pass="${TESTS_PASS:-false}"
  local t_total="${TESTS_TOTAL:-0}"
  local t_passed="${TESTS_PASSED:-0}"
  local t_failed="${TESTS_FAILED:-0}"
  local t_skipped="${TESTS_SKIPPED:-0}"
  local t_dur="${TESTS_DURATION:-0}"
  local s_pass="${START_PASS:-false}"
  local s_dur="${START_DURATION:-0}"
  local s_pid="${START_PID:-0}"
  local h_pass="${HC_PASS:-false}"
  local h_code="${HC_CODE:-0}"
  local h_att="${HC_ATTEMPTS:-0}"
  local h_dur="${HC_DURATION:-0}"
  local p_pass="${STOP_PASS:-false}"
  local p_dur="${STOP_DURATION:-0}"
  local p_orph="${STOP_ORPHANS:-0}"
  cat > "$out" <<EOF
{
  "stack": "$stack",
  "build":  { "pass": $b_pass, "duration_sec": $b_dur, "exit_code": $b_rc },
  "tests":  { "pass": $t_pass, "total": $t_total, "passed": $t_passed, "failed": $t_failed, "skipped": $t_skipped, "duration_sec": $t_dur },
  "start":  { "pass": $s_pass, "duration_sec": $s_dur, "pid": $s_pid },
  "hc":     { "pass": $h_pass, "status_code": $h_code, "attempts": $h_att, "duration_sec": $h_dur },
  "stop":   { "pass": $p_pass, "duration_sec": $p_dur, "orphans": $p_orph }
}
EOF
}

bp_now_sec() { date +%s; }

bp_use_java() {
  local id="$1"
  set +u
  source "$HOME/.sdkman/bin/sdkman-init.sh"
  set -u
  export JAVA_HOME="$HOME/.sdkman/candidates/java/$id"
  if [ ! -d "$JAVA_HOME" ]; then
    set +u
    yes n 2>/dev/null | sdk install java "$id" >/dev/null 2>&1 || true
    set -u
  fi
  export PATH="$JAVA_HOME/bin:$PATH"
}

bp_parse_scalatest() {
  local log="$1"
  local line
  line=$(grep -E 'Tests: succeeded [0-9]+, failed [0-9]+' "$log" | tail -n 1 || true)
  local p f c i pn
  p=$( { echo "$line" | grep -oE 'succeeded [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  f=$( { echo "$line" | grep -oE 'failed [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  c=$( { echo "$line" | grep -oE 'canceled [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  i=$( { echo "$line" | grep -oE 'ignored [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  p=${p:-0}; f=${f:-0}; c=${c:-0}; i=${i:-0}
  local total=$((p + f + c + i))
  echo "TOTAL=$total PASSED=$p FAILED=$f SKIPPED=$((c + i))"
}

bp_parse_surefire() {
  local log="$1"
  local line
  line=$(grep -E 'Tests run: [0-9]+, Failures: [0-9]+, Errors: [0-9]+, Skipped: [0-9]+' "$log" | tail -n 1 || true)
  local n f e s
  n=$( { echo "$line" | grep -oE 'Tests run: [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  f=$( { echo "$line" | grep -oE 'Failures: [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  e=$( { echo "$line" | grep -oE 'Errors: [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  s=$( { echo "$line" | grep -oE 'Skipped: [0-9]+' || true; } | grep -oE '[0-9]+' | head -1)
  n=${n:-0}; f=${f:-0}; e=${e:-0}; s=${s:-0}
  local failed=$((f + e))
  local passed=$((n - failed - s))
  echo "TOTAL=$n PASSED=$passed FAILED=$failed SKIPPED=$s"
}
