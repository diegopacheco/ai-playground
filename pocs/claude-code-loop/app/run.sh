#!/bin/bash
cd "$(dirname "$0")"
go build -o stock-dashboard main.go
./stock-dashboard &
echo $! > app.pid
echo "Started stock-dashboard with PID $(cat app.pid)"
echo "Open http://localhost:8080"
