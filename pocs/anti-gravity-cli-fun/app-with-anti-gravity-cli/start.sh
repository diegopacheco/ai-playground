#!/bin/bash
npm run dev > dev-server.log 2>&1 &
echo $! > dev-server.pid
echo "Server started in the background."
echo "URL: http://localhost:5173/"
