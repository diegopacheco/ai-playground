#!/bin/bash
cd "$(dirname "$0")"

if [ -f .pids ]; then
    while read -r pid; do
        kill "$pid" 2>/dev/null
    done < .pids
    rm .pids
    echo "Stopped all processes"
else
    echo "No .pids file found"
fi
