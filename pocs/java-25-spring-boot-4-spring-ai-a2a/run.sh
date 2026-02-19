#!/bin/bash
mvn clean package -DskipTests

nohup java -jar weather-agent/target/weather-agent-1.0-SNAPSHOT.jar > weather-agent.log 2>&1 &
echo $! > weather-agent.pid

MAX=60
COUNT=0
until curl -s http://localhost:10001/a2a/card > /dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX ]; then
        echo "Weather agent failed to start"
        exit 1
    fi
done
echo "Weather agent running on port 10001"

nohup java -jar hotel-agent/target/hotel-agent-1.0-SNAPSHOT.jar > hotel-agent.log 2>&1 &
echo $! > hotel-agent.pid

COUNT=0
until curl -s http://localhost:10002/a2a/card > /dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX ]; then
        echo "Hotel agent failed to start"
        exit 1
    fi
done
echo "Hotel agent running on port 10002"

nohup java -jar trip-planner/target/trip-planner-1.0-SNAPSHOT.jar > trip-planner.log 2>&1 &
echo $! > trip-planner.pid

COUNT=0
until curl -s http://localhost:10000/a2a/card > /dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX ]; then
        echo "Trip planner failed to start"
        exit 1
    fi
done
echo "Trip planner running on port 10000"
echo "All A2A agents are running"
