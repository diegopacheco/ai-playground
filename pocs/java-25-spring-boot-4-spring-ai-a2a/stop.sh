#!/bin/bash
kill $(cat weather-agent.pid) 2>/dev/null
kill $(cat hotel-agent.pid) 2>/dev/null
kill $(cat trip-planner.pid) 2>/dev/null
rm -f weather-agent.pid hotel-agent.pid trip-planner.pid
echo "All agents stopped"
