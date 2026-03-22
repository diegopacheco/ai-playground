#!/bin/bash

npm install
npm run build

echo ""
echo "Starting openai-oauth proxy in the background..."
npx openai-oauth &
PROXY_PID=$!

for i in $(seq 1 10); do
  curl -s http://127.0.0.1:10531/v1/models > /dev/null 2>&1 && break
  sleep 1
done

echo "Proxy is ready, running the client..."
echo ""
npm run start

kill $PROXY_PID 2>/dev/null
