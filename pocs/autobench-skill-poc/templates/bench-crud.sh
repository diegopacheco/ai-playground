#!/usr/bin/env bash
set -euo pipefail

APP_URL="${1:-http://localhost:8080}"
K6_DURATION="${2:-30s}"
K6_VUS="${3:-50}"
RUNS="${4:-3}"

echo "=== AutoBench: HTTP CRUD Stack ==="
echo "App URL: $APP_URL"
echo "Duration: $K6_DURATION"
echo "Virtual Users: $K6_VUS"
echo "Runs: $RUNS"
echo ""

echo "--- Correctness Check ---"
CREATE_RESP=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$APP_URL/items" \
    -H "Content-Type: application/json" \
    -d '{"name":"bench-test","value":"validation"}')
if [ "$CREATE_RESP" != "201" ] && [ "$CREATE_RESP" != "200" ]; then
    echo "FAIL: POST /items returned $CREATE_RESP (expected 200 or 201)"
    exit 1
fi

READ_RESP=$(curl -s -o /dev/null -w "%{http_code}" "$APP_URL/items")
if [ "$READ_RESP" != "200" ]; then
    echo "FAIL: GET /items returned $READ_RESP (expected 200)"
    exit 1
fi
echo "PASS: CRUD endpoints responding correctly"
echo ""

cat > /tmp/autobench-k6.js << 'SCRIPT'
import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.APP_URL || 'http://localhost:8080';

export default function () {
    let createRes = http.post(`${BASE_URL}/items`, JSON.stringify({
        name: `item-${Date.now()}`,
        value: `val-${Math.random()}`
    }), { headers: { 'Content-Type': 'application/json' } });
    check(createRes, { 'create status 2xx': (r) => r.status >= 200 && r.status < 300 });

    let listRes = http.get(`${BASE_URL}/items`);
    check(listRes, { 'list status 200': (r) => r.status === 200 });

    let id = '1';
    try { id = JSON.parse(createRes.body).id || '1'; } catch(e) {}

    let getRes = http.get(`${BASE_URL}/items/${id}`);
    check(getRes, { 'get status 200': (r) => r.status === 200 });

    let updateRes = http.put(`${BASE_URL}/items/${id}`, JSON.stringify({
        name: `updated-${Date.now()}`,
        value: `updated-${Math.random()}`
    }), { headers: { 'Content-Type': 'application/json' } });
    check(updateRes, { 'update status 2xx': (r) => r.status >= 200 && r.status < 300 });

    let deleteRes = http.del(`${BASE_URL}/items/${id}`);
    check(deleteRes, { 'delete status 2xx': (r) => r.status >= 200 && r.status < 300 });
}
SCRIPT

echo "--- Benchmark Results ---"
TOTAL_RPS=0
TOTAL_P50=0
TOTAL_P95=0
TOTAL_P99=0

for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS:"

    K6_OUTPUT=$(k6 run \
        --vus "$K6_VUS" \
        --duration "$K6_DURATION" \
        --env APP_URL="$APP_URL" \
        --summary-trend-stats="avg,p(50),p(95),p(99)" \
        /tmp/autobench-k6.js 2>&1)

    RPS=$(echo "$K6_OUTPUT" | grep "http_reqs" | awk '{print $2}' | head -1)
    P50=$(echo "$K6_OUTPUT" | grep "http_req_duration" | grep -oP 'p\(50\)=\K[0-9.]+' || echo "0")
    P95=$(echo "$K6_OUTPUT" | grep "http_req_duration" | grep -oP 'p\(95\)=\K[0-9.]+' || echo "0")
    P99=$(echo "$K6_OUTPUT" | grep "http_req_duration" | grep -oP 'p\(99\)=\K[0-9.]+' || echo "0")

    RPS=${RPS:-0}
    P50=${P50:-0}
    P95=${P95:-0}
    P99=${P99:-0}

    echo "  RPS: $RPS"
    echo "  Latency p50: ${P50}ms"
    echo "  Latency p95: ${P95}ms"
    echo "  Latency p99: ${P99}ms"
    echo ""

    TOTAL_RPS=$(echo "$TOTAL_RPS + $RPS" | bc)
    TOTAL_P50=$(echo "$TOTAL_P50 + $P50" | bc)
    TOTAL_P95=$(echo "$TOTAL_P95 + $P95" | bc)
    TOTAL_P99=$(echo "$TOTAL_P99 + $P99" | bc)
done

AVG_RPS=$(echo "scale=2; $TOTAL_RPS / $RUNS" | bc)
AVG_P50=$(echo "scale=2; $TOTAL_P50 / $RUNS" | bc)
AVG_P95=$(echo "scale=2; $TOTAL_P95 / $RUNS" | bc)
AVG_P99=$(echo "scale=2; $TOTAL_P99 / $RUNS" | bc)

echo "--- Average Results ---"
echo "Avg RPS: $AVG_RPS"
echo "Avg Latency p50: ${AVG_P50}ms"
echo "Avg Latency p95: ${AVG_P95}ms"
echo "Avg Latency p99: ${AVG_P99}ms"
echo ""
echo "=== AutoBench Complete ==="
