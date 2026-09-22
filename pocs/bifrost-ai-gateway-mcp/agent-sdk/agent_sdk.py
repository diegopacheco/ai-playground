from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

UNAVAILABLE = "unavailable"
NEVER = "never"
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
SESSION_WINDOW_MINUTES = 360


@dataclass(frozen=True)
class ModelQuota:
    label: str
    remaining: str
    resets_in: str


@dataclass(frozen=True)
class Usage:
    weekly_limit: str = UNAVAILABLE
    current_limit: str = UNAVAILABLE
    time_to_reset_week: str = UNAVAILABLE
    time_to_reset_session: str = UNAVAILABLE
    models: Tuple[ModelQuota, ...] = field(default_factory=tuple)


Runner = Callable[[List[str]], str]


def command_environment(command: str) -> Dict[str, str]:
    environment = dict(os.environ)
    home = Path.home()
    paths = [str(Path(command).parent)] if os.sep in command else []
    paths.extend(str(path) for path in [home / ".local" / "bin", home / ".bun" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")])
    paths.extend(str(path) for path in sorted((home / ".nvm" / "versions" / "node").glob("*/bin"), reverse=True))
    paths.extend(environment.get("PATH", "").split(os.pathsep))
    environment["PATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return environment


def process_runner(command: List[str]) -> str:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, env=command_environment(command[0]))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {(result.stderr or result.stdout).strip()}")
    return result.stdout


def json_object(output: str) -> Dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def format_duration(millis: int) -> str:
    if millis <= 0:
        return "0m"
    total_minutes = millis // 60000
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"


def format_percent(value: float) -> str:
    return f"{round(max(0.0, min(100.0, value)))}%"


def duration_until(moment: object) -> str:
    reset = None
    if isinstance(moment, bool):
        return UNAVAILABLE
    if isinstance(moment, (int, float)):
        reset = datetime.fromtimestamp(moment, timezone.utc)
    elif isinstance(moment, str) and moment:
        try:
            reset = datetime.fromisoformat(moment.replace("Z", "+00:00"))
        except ValueError:
            return UNAVAILABLE
    if reset is None:
        return UNAVAILABLE
    if reset.tzinfo is None:
        reset = reset.replace(tzinfo=timezone.utc)
    return format_duration(int((reset - datetime.now(timezone.utc)).total_seconds() * 1000))


class Agent:
    def __init__(self, runner: Runner = process_runner):
        self._runner = runner

    def call(self, model: str, prompt: str, args: Optional[List[str]] = None) -> str:
        if not model.strip():
            raise ValueError("model is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        return self._runner(self._command(model, prompt, list(args or [])))

    def usage(self) -> Usage:
        return Usage()

    def _command(self, model: str, prompt: str, args: List[str]) -> List[str]:
        raise NotImplementedError


def curl_config(token: str) -> str:
    headers = [f"Authorization: Bearer {token}", "anthropic-beta: oauth-2025-04-20", "Accept: application/json"]
    return "".join([f'url = "{USAGE_URL}"\n', *[f'header = "{header}"\n' for header in headers]])


def access_token(credentials: Dict[str, Any]) -> str:
    oauth = credentials.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return ""
    token = oauth.get("accessToken")
    return token if isinstance(token, str) else ""


def parse_claude_usage(data: Dict[str, Any]) -> Usage:
    values: Dict[str, str] = {}
    for key, prefix in (("five_hour", "session"), ("seven_day", "week")):
        window = data.get(key)
        if not isinstance(window, dict):
            continue
        utilization = window.get("utilization")
        if not isinstance(utilization, (int, float)):
            continue
        values[f"{prefix}_limit"] = format_percent(100.0 - float(utilization))
        values[f"{prefix}_reset"] = duration_until(window.get("resets_at"))
    return Usage(
        values.get("week_limit", UNAVAILABLE),
        values.get("session_limit", UNAVAILABLE),
        values.get("week_reset", UNAVAILABLE),
        values.get("session_reset", UNAVAILABLE),
    )


class ClaudeCodeAgent(Agent):
    def _command(self, model: str, prompt: str, args: List[str]) -> List[str]:
        return ["claude", "-p", "--model", model, *args, prompt]

    def usage(self) -> Usage:
        token = self._token()
        if not token:
            return Usage()
        try:
            return parse_claude_usage(json_object(self._read_usage(token)))
        except Exception:
            return Usage()

    def _read_usage(self, token: str) -> str:
        handle, path = tempfile.mkstemp(prefix="agent-sdk-", suffix=".curl")
        try:
            os.write(handle, curl_config(token).encode("utf-8"))
            os.close(handle)
            return self._runner(["curl", "-sS", "-K", path])
        finally:
            Path(path).unlink(missing_ok=True)

    def _token(self) -> str:
        configured = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if configured:
            return configured
        try:
            token = access_token(json_object(self._runner(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"])))
            if token:
                return token
        except Exception:
            pass
        home = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
        try:
            return access_token(json_object((home / ".credentials.json").read_text(encoding="utf-8")))
        except OSError:
            return ""


def latest_rate_limits(home: Path) -> Dict[str, Any]:
    root = home / "sessions"
    if not root.is_dir():
        return {}
    try:
        files = sorted(root.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)[:20]
    except OSError:
        return {}
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            payload = json_object(line).get("payload")
            if not isinstance(payload, dict):
                continue
            limits = payload.get("rate_limits")
            if isinstance(limits, dict) and (isinstance(limits.get("primary"), dict) or isinstance(limits.get("secondary"), dict)):
                return limits
    return {}


def parse_codex_usage(limits: Dict[str, Any]) -> Usage:
    values: Dict[str, str] = {}
    for window in (limits.get("primary"), limits.get("secondary")):
        if not isinstance(window, dict):
            continue
        used = window.get("used_percent")
        if not isinstance(used, (int, float)):
            continue
        minutes = window.get("window_minutes")
        prefix = "session" if isinstance(minutes, (int, float)) and minutes <= SESSION_WINDOW_MINUTES else "week"
        values[f"{prefix}_limit"] = format_percent(100.0 - float(used))
        values[f"{prefix}_reset"] = duration_until(window.get("resets_at"))
    return Usage(
        values.get("week_limit", UNAVAILABLE),
        values.get("session_limit", UNAVAILABLE),
        values.get("week_reset", UNAVAILABLE),
        values.get("session_reset", UNAVAILABLE),
    )


class CodexAgent(Agent):
    def _command(self, model: str, prompt: str, args: List[str]) -> List[str]:
        return ["codex", "exec", "--model", model, *args, prompt]

    def usage(self) -> Usage:
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        return parse_codex_usage(latest_rate_limits(home))


def parse_agy_usage(data: Dict[str, Any]) -> Usage:
    models = data.get("models")
    if not isinstance(models, list):
        return Usage()
    quotas: Dict[str, Tuple[float, int]] = {}
    for value in models:
        if not isinstance(value, dict) or value.get("isAutocompleteOnly") is True:
            continue
        remaining = value.get("remainingPercentage")
        label = value.get("label") or value.get("modelId")
        if not isinstance(remaining, (int, float)) or not isinstance(label, str) or not label:
            continue
        reset = value.get("timeUntilResetMs")
        candidate = (float(remaining), int(reset) if isinstance(reset, (int, float)) else 0)
        current = quotas.get(label)
        if current is None or candidate[0] < current[0]:
            quotas[label] = candidate
    if not quotas:
        return Usage()
    models_quota = tuple(ModelQuota(label, format_percent(remaining * 100), format_duration(reset)) for label, (remaining, reset) in quotas.items())
    lowest = min(quotas, key=lambda name: quotas[name][0])
    remaining, reset = quotas[lowest]
    return Usage(UNAVAILABLE, format_percent(remaining * 100), UNAVAILABLE, format_duration(reset), models_quota)


class AgyAgent(Agent):
    def _command(self, model: str, prompt: str, args: List[str]) -> List[str]:
        return ["agy", "--model", model, *args, "-p", prompt]

    def usage(self) -> Usage:
        try:
            output = self._runner(["antigravity-usage", "quota", "--json"])
        except Exception:
            return Usage()
        return parse_agy_usage(json_object(output))


class OllamaAgent(Agent):
    def _command(self, model: str, prompt: str, args: List[str]) -> List[str]:
        return ["ollama", "run", "--nowordwrap", *args, model, prompt]

    def usage(self) -> Usage:
        return Usage("100%", "100%", NEVER, NEVER)
