source "$(dirname "$0")/scripts/common.sh"
result=0
for service in api frontend; do
  if running "$service"; then
    echo "$service running (PID $pid)."
  else
    echo "$service stopped."
    result=1
  fi
done
exit "$result"
