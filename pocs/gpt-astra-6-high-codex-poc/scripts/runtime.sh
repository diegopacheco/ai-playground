#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
pid_file="$project_root/.runtime/server.pid"
log_file="$project_root/.runtime/server.log"
server_pid=""
is_running() {
  [[ -f "$pid_file" ]] || return 1
  read -r server_pid < "$pid_file" || return 1
  [[ "$server_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$server_pid" 2>/dev/null || return 1
  local command
  command="$(ps -p "$server_pid" -o command=)" || return 1
  [[ "$command" == "node $project_root/server.js $project_root/dist" ]]
}
