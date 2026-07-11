#!/usr/bin/env bash
nohup python3 -m http.server 8080 --bind 127.0.0.1 > /tmp/pacificarium.log 2>&1 &
echo $! > /tmp/pacificarium.pid
echo "Pacificarium running at http://127.0.0.1:8080"
