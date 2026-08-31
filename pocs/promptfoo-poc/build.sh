#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "installing dependencies"
npm install

echo "checking ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed, get it from https://ollama.com"
  exit 1
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "starting ollama"
  ollama serve >/dev/null 2>&1 &
  for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null; then break; fi
    sleep 1
  done
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "ollama is not reachable on localhost:11434"
  exit 1
fi

for model in llama3.2 qwen2.5-coder; do
  if ollama list | grep -q "^$model"; then
    echo "model $model present"
  else
    echo "pulling model $model"
    ollama pull "$model"
  fi
done

echo "build ok"
