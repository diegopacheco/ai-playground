#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="$HOME/.claude-hooks/tool-time-tracker.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "No log file found at $LOG_FILE"
    exit 0
fi

ENTRIES=$(wc -l < "$LOG_FILE" | tr -d ' ')
FILE_SIZE=$(du -h "$LOG_FILE" | cut -f1)

echo "=== CC Tool Time Tracker Report ==="
echo "Log file: $LOG_FILE"
echo "Entries: $ENTRIES | Size: $FILE_SIZE"
echo ""

echo "--- Per-Tool Summary (sorted by total time) ---"
printf "%-20s %8s %10s %10s %10s %12s\n" "TOOL" "COUNT" "AVG(ms)" "MIN(ms)" "MAX(ms)" "TOTAL(ms)"
printf "%-20s %8s %10s %10s %10s %12s\n" "----" "-----" "-------" "-------" "-------" "---------"

jq -r '.tool + " " + (.elapsed_ms | tostring)' "$LOG_FILE" | \
    awk '{
        tool=$1; ms=$2;
        count[tool]++;
        total[tool]+=ms;
        if (!(tool in min_val) || ms < min_val[tool]) min_val[tool]=ms;
        if (ms > max_val[tool]) max_val[tool]=ms;
    }
    END {
        for (t in count) {
            printf "%-20s %8d %10d %10d %10d %12d\n", t, count[t], total[t]/count[t], min_val[t], max_val[t], total[t]
        }
    }' | sort -t' ' -k6 -rn

echo ""
echo "--- Last 10 Entries ---"
tail -10 "$LOG_FILE" | jq -r '"\(.timestamp) | \(.tool) | \(.elapsed_ms)ms"'
