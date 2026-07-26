from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import config


@dataclass(frozen=True)
class AgentSpec:
    key: str
    label: str
    binary: str
    default_model: str

    def command(self, prompt: str, model: str) -> list[str]:
        if self.key == "claude":
            return [self.binary, "-p", "--model", model, prompt]
        if self.key == "codex":
            return [self.binary, "exec", "--skip-git-repo-check", "-m", model, prompt]
        return [self.binary, "-p", prompt, "--model", model]


SPECS: dict[str, AgentSpec] = {
    "claude": AgentSpec("claude", "Claude Opus 5", "claude", "opus"),
    "codex": AgentSpec("codex", "Codex 5.6-sol", "codex", "gpt-5.6-sol"),
    "gemini": AgentSpec("gemini", "Gemini (agy)", "agy", "gemini-3-pro"),
}

DEFAULT_PREFERENCES: dict[str, Any] = {
    "active": "claude",
    "models": {key: spec.default_model for key, spec in SPECS.items()},
    "timeout": 180,
}


class AgentPreferences:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        if not config.agents_file.is_file():
            return json.loads(json.dumps(DEFAULT_PREFERENCES))
        stored: dict[str, Any] = json.loads(config.agents_file.read_text())
        merged = json.loads(json.dumps(DEFAULT_PREFERENCES))
        merged["active"] = stored.get("active", merged["active"])
        merged["timeout"] = stored.get("timeout", merged["timeout"])
        merged["models"].update(stored.get("models", {}))
        return merged

    def values(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._values))

    def update(self, active: str | None, models: dict[str, str] | None, timeout: int | None) -> dict[str, Any]:
        with self._lock:
            if active is not None:
                if active not in SPECS:
                    raise ValueError(f"unknown agent: {active}")
                self._values["active"] = active
            if models:
                for key, model in models.items():
                    if key not in SPECS:
                        raise ValueError(f"unknown agent: {key}")
                    self._values["models"][key] = model
            if timeout is not None:
                self._values["timeout"] = max(10, min(timeout, 900))
            config.agents_file.write_text(json.dumps(self._values, indent=2))
            return self.values()


preferences: AgentPreferences = AgentPreferences()


def availability() -> list[dict[str, Any]]:
    values = preferences.values()
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "binary": spec.binary,
            "model": values["models"][spec.key],
            "default_model": spec.default_model,
            "installed": shutil.which(spec.binary) is not None,
        }
        for spec in SPECS.values()
    ]


def ask(prompt: str, agent_key: str | None = None) -> dict[str, Any]:
    values = preferences.values()
    key = agent_key or values["active"]
    spec = SPECS.get(key)
    if spec is None:
        raise ValueError(f"unknown agent: {key}")
    if shutil.which(spec.binary) is None:
        raise RuntimeError(f"{spec.binary} is not installed or not on PATH")

    model = values["models"][key]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            spec.command(prompt, model),
            capture_output=True,
            text=True,
            timeout=values["timeout"],
            cwd=str(config.data_dir),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{spec.label} timed out after {values['timeout']}s")

    elapsed = round(time.monotonic() - started, 2)
    answer = completed.stdout.strip()
    if completed.returncode != 0 and not answer:
        raise RuntimeError(completed.stderr.strip() or f"{spec.label} exited {completed.returncode}")

    return {
        "agent": key,
        "label": spec.label,
        "model": model,
        "answer": answer,
        "elapsed_seconds": elapsed,
        "exit_code": completed.returncode,
    }
