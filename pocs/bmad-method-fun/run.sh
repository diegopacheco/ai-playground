#!/bin/sh
set -e
cd backend
cargo run &
backend_pid=$!
cd ../frontend
bun run dev &
frontend_pid=$!
trap 'kill $backend_pid $frontend_pid' INT TERM EXIT
wait
