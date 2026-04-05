#!/bin/bash
cd "$(dirname "$0")"
bun install
bun run dev &
echo $! > .frontend.pid
echo "Frontend started on http://localhost:5173 (PID: $(cat .frontend.pid))"
