#!/bin/bash
cd "$(dirname "$0")"
cargo build --release 2>&1
./target/release/pixel-office-backend &
echo $! > .backend.pid
echo "Backend started on http://localhost:3001 (PID: $(cat .backend.pid))"
