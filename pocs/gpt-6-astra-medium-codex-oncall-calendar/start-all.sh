source "$(dirname "$0")/scripts/common.sh"
bash setup.sh
mkdir -p "$run_dir"
started=""
cleanup() {
  for service in $started; do
    if running "$service"; then kill "$pid"; fi
    rm -f "$run_dir/$service.pid"
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for service in api frontend; do
  if running "$service"; then
    echo "$service already running (PID $pid)."
    continue
  fi
  rm -f "$run_dir/$service.pid"
  if test "$service" = api; then port=8000; else port=5173; fi
  python3.14 - "$port" <<'PY'
import socket
import sys

with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as error:
        sys.exit(f"Cannot bind port {sys.argv[1]}: {error}")
PY
  if test "$service" = api; then
    nohup python3.14 "$project_dir/backend/server.py" >"$run_dir/api.log" 2>&1 </dev/null &
    url="http://127.0.0.1:8000/api/health"
  else
    nohup bun "$project_dir/node_modules/vite/bin/vite.js" --host 127.0.0.1 --port 5173 --strictPort >"$run_dir/frontend.log" 2>&1 </dev/null &
    url="http://127.0.0.1:5173"
  fi
  echo "$!" >"$run_dir/$service.pid"
  started="$started $service"
  ready=false
  for ((attempt=0; attempt<15; attempt++)); do
    running "$service" || break
    if curl --fail --silent --max-time 1 "$url" >/dev/null; then ready=true; break; fi
    sleep 1
  done
  if test "$ready" = false; then
    echo "$service failed to start. See $run_dir/$service.log" >&2
    cat "$run_dir/$service.log" >&2
    exit 1
  fi
  echo "$service started (PID $pid)."
done
trap - EXIT INT TERM
echo "Open http://127.0.0.1:5173. Logs: $run_dir"
