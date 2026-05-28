#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/app"

if [ ! -f .env ]; then
  if [ -f env.example ]; then
    cp env.example .env
    echo "Created app/.env from env.example - add your EXPO_PUBLIC_OPENAI_API_KEY"
  fi
fi

if [ ! -d node_modules ]; then
  npm install
fi

echo "Starting Expo. Scan the QR below with Expo Go (Android) or the Camera app (iOS)."
echo "Phone and this computer must be on the same Wi-Fi. Press Ctrl+C to stop."
exec npx expo start
