from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import config


def status() -> dict[str, Any]:
    return {
        "binary": str(config.rust_binary),
        "binary_ready": config.rust_binary.is_file(),
        "model": str(config.gguf_model),
        "model_ready": config.gguf_model.is_file(),
    }


def run(pdf_path: Path, question: str, timeout: int = 600) -> dict[str, Any]:
    if not config.rust_binary.is_file():
        raise RuntimeError(f"rust binary missing: {config.rust_binary} (run ./build.sh)")
    if not config.gguf_model.is_file():
        raise RuntimeError(f"gguf model missing: {config.gguf_model}")

    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(config.rust_binary), str(pdf_path), question, str(config.gguf_model)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"rust llama timed out after {timeout}s")

    elapsed = round(time.monotonic() - started, 2)
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    try:
        payload: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        raise RuntimeError(completed.stderr.strip()[-800:] or "rust llama produced no output")

    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error", "rust llama failed")))

    payload["elapsed_seconds"] = elapsed
    return payload
