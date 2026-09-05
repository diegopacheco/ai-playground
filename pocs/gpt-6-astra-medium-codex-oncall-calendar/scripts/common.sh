set -eu
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
run_dir="$project_dir/.run"

running() {
  test -f "$run_dir/$1.pid" || return 1
  pid="$(cat "$run_dir/$1.pid")"
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  test "$pid" -gt 1 || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command_line="$(ps -p "$pid" -o command=)" || return 1
  case "$1:$command_line" in
    api:*"$project_dir/backend/server.py"*|frontend:*"$project_dir/node_modules/vite/bin/vite.js"*) return 0 ;;
    *) return 1 ;;
  esac
}
