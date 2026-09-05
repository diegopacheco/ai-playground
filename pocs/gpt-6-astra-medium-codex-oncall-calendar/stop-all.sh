source "$(dirname "$0")/scripts/common.sh"
result=0
for service in frontend api; do
  if running "$service"; then
    kill "$pid" 2>/dev/null || true
    for ((attempt=0; attempt<10; attempt++)); do
      running "$service" || break
      sleep 1
    done
    if running "$service"; then
      echo "$service did not stop (PID $pid)." >&2
      result=1
      continue
    fi
    echo "$service stopped."
  else
    echo "$service is not running."
  fi
  rm -f "$run_dir/$service.pid"
done
exit "$result"
