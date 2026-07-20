#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
runner_dir="/tmp/flowprint"
runner_pid_file="$runner_dir/runner.pid"
runner_log_file="$runner_dir/runner.log"
mkdir -p "$runner_dir"
if [[ -f "$runner_pid_file" ]]; then
  runner_pid="$(<"$runner_pid_file")"
  if kill -0 "$runner_pid" 2>/dev/null; then
    if cmp -s "$project_dir/src/runner.py" "$runner_dir/runner.py"; then
      exit 0
    fi
    "$project_dir/stop-runner.sh"
  fi
fi
cp "$project_dir/src/runner.py" "$runner_dir/runner.py"
nohup python3 -B "$runner_dir/runner.py" "$project_dir" >"$runner_log_file" 2>&1 &
runner_pid="$!"
printf '%s\n' "$runner_pid" >"$runner_pid_file"
for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:17339/status >/dev/null 2>&1; then
    printf 'FlowPrint runner started.\n'
    exit 0
  fi
  if ! kill -0 "$runner_pid" 2>/dev/null; then
    cat "$runner_log_file"
    exit 1
  fi
  sleep 1
done
printf 'FlowPrint runner did not start.\n'
exit 1
