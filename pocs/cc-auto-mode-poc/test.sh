#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/app"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "$EXPO_PUBLIC_OPENAI_API_KEY" ]; then
  echo "Set EXPO_PUBLIC_OPENAI_API_KEY in app/.env or the environment first"
  exit 1
fi

node test/test-openai.mjs
