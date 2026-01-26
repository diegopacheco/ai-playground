#!/bin/bash
cd "$(dirname "$0")"
echo "Stopping any running agent-learner processes..."
pkill -f "agent-learner" 2>/dev/null
pkill -f "claude.*dangerously-skip-permissions" 2>/dev/null
echo "Stopped"
