import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeIs

type CommandRunner = Callable[[list[str]], str]
type Provider = Literal["claude", "codex", "agy", "ollama"]

PROVIDERS: tuple[Provider, ...] = ("claude", "codex", "agy", "ollama")


class AgentError(Exception):
    pass


def is_provider(value: str) -> TypeIs[Provider]:
    return value in PROVIDERS


def node_binaries(home: Path) -> list[str]:
    root = home / ".nvm" / "versions" / "node"
    if not root.is_dir():
        return []
    return [str(version / "bin") for version in sorted(root.iterdir(), reverse=True)]


def command_path(command: str) -> str:
    home = Path.home()
    candidates = [
        *([str(Path(command).parent)] if "/" in command else []),
        str(home / ".local" / "bin"),
        str(home / ".bun" / "bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        *node_binaries(home),
        *os.environ.get("PATH", "").split(os.pathsep),
    ]
    return os.pathsep.join(dict.fromkeys(item for item in candidates if item))


def process_runner(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PATH": command_path(command[0])},
        )
    except FileNotFoundError as error:
        raise AgentError(f"Unable to start {command[0]}: {error}") from error
    if result.returncode != 0:
        raise AgentError(f"Command failed with exit code {result.returncode}: {(result.stderr or result.stdout).strip()}")
    return result.stdout


class Agent(ABC):
    def __init__(self, runner: CommandRunner = process_runner) -> None:
        self.runner = runner

    def call(self, model: str, prompt: str, args: Sequence[str] = ()) -> str:
        if not model.strip():
            raise ValueError("model is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        return self.runner(self.command(model, prompt, list(args)))

    @abstractmethod
    def command(self, model: str, prompt: str, args: list[str]) -> list[str]: ...


class ClaudeCodeAgent(Agent):
    def command(self, model: str, prompt: str, args: list[str]) -> list[str]:
        return ["claude", "-p", "--model", model, *args, prompt]


class CodexAgent(Agent):
    def command(self, model: str, prompt: str, args: list[str]) -> list[str]:
        return ["codex", "exec", "--model", model, *args, prompt]


class AgyAgent(Agent):
    def command(self, model: str, prompt: str, args: list[str]) -> list[str]:
        return ["agy", "-p", "--model", model, *args, prompt]


class OllamaAgent(Agent):
    def command(self, model: str, prompt: str, args: list[str]) -> list[str]:
        return ["ollama", "run", *args, model, prompt]


AGENTS: dict[Provider, type[Agent]] = {
    "claude": ClaudeCodeAgent,
    "codex": CodexAgent,
    "agy": AgyAgent,
    "ollama": OllamaAgent,
}


def agent_for(provider: Provider, runner: CommandRunner = process_runner) -> Agent:
    return AGENTS[provider](runner)
