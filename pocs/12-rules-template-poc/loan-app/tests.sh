#!/usr/bin/env bash
cd "$(dirname "$0")"
mkdir -p .run

FAILED=0
RESULTS=()

mark_pass() { RESULTS+=("PASS  $1"); }
mark_fail() { RESULTS+=("FAIL  $1"); FAILED=1; }

cleanup() {
  if [ -n "${BACKEND_PID:-}" ]; then
    pkill -P "$BACKEND_PID" 2>/dev/null || true
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID:-}" ]; then
    pkill -P "$FRONTEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  pkill -f spring-boot:run 2>/dev/null || true
  pkill -f com.loan.LoanApplication 2>/dev/null || true
  pkill -f "vite" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Backend (JUnit 5: unit + integration)"
if (cd backend && mvn -B test); then
  mark_pass "Backend (JUnit 5)"
else
  mark_fail "Backend (JUnit 5)"
fi

echo
echo "==> Frontend (Jest)"
mkdir -p frontend/test-results
(cd frontend && bun install --silent)
if (cd frontend && bun run test); then
  mark_pass "Frontend (Jest)"
else
  mark_fail "Frontend (Jest)"
fi

echo
echo "==> Starting backend + frontend for e2e and stress"
(cd backend && mvn -q spring-boot:run > ../.run/backend-tests.log 2>&1) &
BACKEND_PID=$!
(cd frontend && bun run dev > ../.run/frontend-tests.log 2>&1) &
FRONTEND_PID=$!

echo "    waiting for backend GraphQL on :8080..."
n=0
until curl -sf -X POST http://localhost:8080/graphql \
    -H 'content-type: application/json' \
    -d '{"query":"{health}"}' >/dev/null 2>&1; do
  n=$((n+1))
  if [ $n -ge 120 ]; then
    echo "    backend did not become ready within 120s"
    mark_fail "Backend startup"
    break
  fi
  sleep 1
done

echo "    waiting for frontend on :5173..."
n=0
until curl -sf http://localhost:5173 >/dev/null 2>&1; do
  n=$((n+1))
  if [ $n -ge 60 ]; then
    echo "    frontend did not become ready within 60s"
    mark_fail "Frontend startup"
    break
  fi
  sleep 1
done

echo
echo "==> E2E (Playwright)"
(cd e2e && bun install --silent)
(cd e2e && bunx playwright install chromium > /dev/null 2>&1) || true
if (cd e2e && bun run test); then
  mark_pass "E2E (Playwright)"
else
  mark_fail "E2E (Playwright)"
fi

echo
echo "==> Stress (k6)"
if k6 run --summary-export=.run/k6-summary.json stress/loan.js; then
  mark_pass "Stress (k6)"
else
  mark_fail "Stress (k6)"
fi

echo
echo "==> Summary"
for r in "${RESULTS[@]}"; do
  echo "  $r"
done

if [ $FAILED -eq 0 ]; then
  echo
  echo "ALL TESTS PASSED"
else
  echo
  echo "SOME TESTS FAILED"
fi

echo "run ./report.sh to render the aggregate report"
exit $FAILED
