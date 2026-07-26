#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[1/3] python backend"
if [ ! -d backend/.venv ]; then
  python3.14 -m venv backend/.venv
fi
backend/.venv/bin/python -m pip install -q --upgrade pip
backend/.venv/bin/pip install -q \
  fastapi "uvicorn[standard]" python-multipart \
  llama-index-core llama-index-readers-file \
  llama-index-embeddings-ollama llama-index-llms-ollama pypdf

echo "[2/3] rust pdfllama"
cd "$ROOT/rust"
cargo build --release
cd "$ROOT"

echo "[3/3] frontend"
cd "$ROOT/frontend"
npm install
cd "$ROOT"

echo "build done"
