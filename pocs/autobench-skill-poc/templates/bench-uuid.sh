#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-http://localhost:8080}"
CONCURRENCY="${2:-50}"
TOTAL_REQUESTS="${3:-10000}"
RUNS="${4:-3}"

echo "=== AutoBench: WebServer UUID ==="
echo "Server URL: $SERVER_URL"
echo "Concurrency: $CONCURRENCY"
echo "Total Requests: $TOTAL_REQUESTS"
echo "Runs: $RUNS"
echo ""

echo "--- Correctness Check ---"
UUIDS=""
for j in $(seq 1 10); do
    UUID=$(curl -s "$SERVER_URL/uuid" | grep -oP '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' || echo "")
    if [ -z "$UUID" ]; then
        echo "FAIL: Response does not contain a valid UUID"
        exit 1
    fi
    UUIDS="$UUIDS\n$UUID"
done

UNIQUE_COUNT=$(echo -e "$UUIDS" | sort -u | grep -c '[0-9a-f]')
if [ "$UNIQUE_COUNT" -lt 10 ]; then
    echo "FAIL: UUIDs are not unique ($UNIQUE_COUNT unique out of 10)"
    exit 1
fi
echo "PASS: All UUIDs valid and unique"
echo ""

echo "--- Benchmark Results ---"
TOTAL_RPS=0
TOTAL_AVG_LATENCY=0
TOTAL_P99_LATENCY=0

for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS:"

    if command -v hey &> /dev/null; then
        OUTPUT=$(hey -n "$TOTAL_REQUESTS" -c "$CONCURRENCY" "$SERVER_URL/uuid" 2>&1)
        RPS=$(echo "$OUTPUT" | grep "Requests/sec" | awk '{print $2}')
        AVG_LAT=$(echo "$OUTPUT" | grep "Average" | head -1 | awk '{print $2}')
        P99_LAT=$(echo "$OUTPUT" | grep "99%" | awk '{print $2}' | head -1)
    elif command -v wrk &> /dev/null; then
        OUTPUT=$(wrk -t4 -c"$CONCURRENCY" -d10s "$SERVER_URL/uuid" 2>&1)
        RPS=$(echo "$OUTPUT" | grep "Req/Sec" | awk '{print $2}' | tr -d 'k' | head -1)
        AVG_LAT=$(echo "$OUTPUT" | grep "Latency" | awk '{print $2}' | head -1)
        P99_LAT=$(echo "$OUTPUT" | grep "99%" | awk '{print $2}' | head -1)
    else
        START_TIME=$(date +%s%N)
        for r in $(seq 1 $TOTAL_REQUESTS); do
            curl -s -o /dev/null "$SERVER_URL/uuid" &
            if [ $((r % CONCURRENCY)) -eq 0 ]; then
                wait
            fi
        done
        wait
        END_TIME=$(date +%s%N)
        ELAPSED_S=$(echo "scale=3; ($END_TIME - $START_TIME) / 1000000000" | bc)
        RPS=$(echo "scale=2; $TOTAL_REQUESTS / $ELAPSED_S" | bc)
        AVG_LAT=$(echo "scale=3; 1000 / $RPS * $CONCURRENCY" | bc)
        P99_LAT="N/A"
    fi

    RPS=${RPS:-0}
    AVG_LAT=${AVG_LAT:-0}
    P99_LAT=${P99_LAT:-0}

    echo "  RPS: $RPS"
    echo "  Avg Latency: ${AVG_LAT}ms"
    echo "  p99 Latency: ${P99_LAT}ms"
    echo ""

    TOTAL_RPS=$(echo "$TOTAL_RPS + $RPS" | bc 2>/dev/null || echo "$TOTAL_RPS")
    TOTAL_AVG_LATENCY=$(echo "$TOTAL_AVG_LATENCY + ${AVG_LAT//ms/}" | bc 2>/dev/null || echo "$TOTAL_AVG_LATENCY")
    TOTAL_P99_LATENCY=$(echo "$TOTAL_P99_LATENCY + ${P99_LAT//ms/}" | bc 2>/dev/null || echo "$TOTAL_P99_LATENCY")
done

AVG_RPS=$(echo "scale=2; $TOTAL_RPS / $RUNS" | bc 2>/dev/null || echo "N/A")
FINAL_AVG_LAT=$(echo "scale=3; $TOTAL_AVG_LATENCY / $RUNS" | bc 2>/dev/null || echo "N/A")
FINAL_P99_LAT=$(echo "scale=3; $TOTAL_P99_LATENCY / $RUNS" | bc 2>/dev/null || echo "N/A")

echo "--- Average Results ---"
echo "Avg RPS: $AVG_RPS"
echo "Avg Latency: ${FINAL_AVG_LAT}ms"
echo "Avg p99 Latency: ${FINAL_P99_LAT}ms"
echo ""
echo "=== AutoBench Complete ==="
