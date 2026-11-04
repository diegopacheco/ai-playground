import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMMAND_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class SuiteRun:
    suite: str
    cases: list[dict[str, Any]]


def run_command(command: list[str], cwd: Path, stdin: str) -> tuple[str, int]:
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS
    )
    latency = int((time.monotonic() - started) * 1000)
    output = completed.stdout if completed.returncode == 0 else completed.stdout + completed.stderr
    return output.strip(), latency


def load_suite(path: Path) -> SuiteRun:
    data = json.loads(path.read_text())
    cases: list[dict[str, Any]] = []
    for raw in data["cases"]:
        case = {"name": raw["name"], "input": raw.get("input", ""), "checks": raw["checks"]}
        if "command" in raw:
            command = [str(part) for part in raw["command"]]
            output, latency = run_command(command, path.parent, str(raw.get("stdin", "")))
            case["input"] = case["input"] or " ".join(command)
            case["output"], case["latency_ms"] = output, latency
        elif "output_file" in raw:
            case["output"] = (path.parent / raw["output_file"]).read_text()
            case["latency_ms"] = 0
        else:
            case["output"] = str(raw.get("output", ""))
            case["latency_ms"] = int(raw.get("latency_ms", 0))
        cases.append(case)
    return SuiteRun(str(data["suite"]), cases)
