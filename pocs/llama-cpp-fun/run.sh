#!/usr/bin/env bash
set -euo pipefail

required_python="3.14.6"
python_bin="${PYTHON_BIN:-python3.14}"
llama_bin="${LLAMA_BIN:-llama}"
model="ggml-org/Qwen3.6-27B-GGUF:Q8_0"
port="${LLAMA_CPP_PORT:-8081}"

if ! command -v "$python_bin" >/dev/null; then
    printf '%s was not found\n' "$python_bin" >&2
    exit 1
fi

if ! command -v "$llama_bin" >/dev/null; then
    printf '%s was not found\n' "$llama_bin" >&2
    exit 1
fi

actual_python="$($python_bin -c 'import platform; print(platform.python_version())')"
if [[ "$actual_python" != "$required_python" ]]; then
    printf 'Python %s is required, found %s\n' "$required_python" "$actual_python" >&2
    exit 1
fi

if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; then
    LLAMA_CPP_BASE_URL="http://127.0.0.1:$port/v1" "$python_bin" main.py "$@"
    exit 0
fi

"$llama_bin" serve -hf "$model" --host 127.0.0.1 --port "$port" --n-gpu-layers 99 &
server_pid=$!

stop_server() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}

trap stop_server EXIT INT TERM

until curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
    fi
    sleep 1
done

LLAMA_CPP_BASE_URL="http://127.0.0.1:$port/v1" "$python_bin" main.py "$@"
