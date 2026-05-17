#!/usr/bin/env bash
cd "$(dirname "$0")"

for s in backend frontend; do
  if [ -f .run/$s.pid ]; then
    pid=$(cat .run/$s.pid)
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
    rm -f .run/$s.pid
  fi
done

pkill -f "spring-boot:run" 2>/dev/null || true
pkill -f "com.loan.LoanApplication" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

echo "stopped"
