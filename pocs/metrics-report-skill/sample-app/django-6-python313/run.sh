#!/bin/bash

cd backend
../venv/bin/python manage.py runserver 8080 &
BACKEND_PID=$!
echo $BACKEND_PID > ../backend.pid

cd ../frontend
npm start &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../frontend.pid

echo "Backend PID: $BACKEND_PID (port 8080)"
echo "Frontend PID: $FRONTEND_PID (port 3000)"
echo "PIDs saved to backend.pid and frontend.pid"

wait
