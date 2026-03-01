#!/bin/bash
cd db && bash start-db.sh
cd ../backend && bash run.sh &
BACKEND_PID=$!
while ! curl -s http://localhost:3000/api/leaderboard > /dev/null 2>&1; do
  sleep 1
done
cd ../frontend && bash run.sh &
FRONTEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend:  http://localhost:3000"
echo "Frontend: http://localhost:5173"
wait
