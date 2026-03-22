#!/bin/bash
PASS=0
FAIL=0
check() {
  if [ "$1" = "200" ]; then
    echo "PASS: $2"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $2 (got $1)"
    FAIL=$((FAIL + 1))
  fi
}
GRAFANA=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health 2>/dev/null)
check "$GRAFANA" "Grafana health"
PROM=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/-/healthy 2>/dev/null)
check "$PROM" "Prometheus health"
DASH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/dashboards/uid/agent-terminator-game-grafana 2>/dev/null)
check "$DASH" "Dashboard loaded"
DS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/datasources 2>/dev/null)
check "$DS" "Datasources configured"
echo ""
echo "Results: $PASS passed, $FAIL failed"
