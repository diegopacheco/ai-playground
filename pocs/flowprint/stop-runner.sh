#!/usr/bin/env bash
set -euo pipefail
runner_pid_file="/tmp/flowprint/runner.pid"
if [[ ! -f "$runner_pid_file" ]]; then
  exit 0
fi
runner_pid="$(<"$runner_pid_file")"
if kill -0 "$runner_pid" 2>/dev/null; then
  kill "$runner_pid"
  for _ in {1..30}; do
    if ! kill -0 "$runner_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f "$runner_pid_file"
printf 'FlowPrint runner stopped.\n'
