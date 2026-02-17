#!/bin/bash

cd "$(dirname "$0")"

cargo build --release
./target/release/main &
echo $! > .backend.pid

cd frontend
npm run dev &
echo $! > ../.frontend.pid
cd ..

echo "Backend running at http://localhost:3001"
echo "Frontend running at http://localhost:5173"
