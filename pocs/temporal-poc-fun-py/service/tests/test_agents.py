import pytest

from agent_service.agents import AgentError, agent_for, command_path, is_provider, process_runner


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("claude", ["claude", "-p", "--model", "m", "hi"]),
        ("codex", ["codex", "exec", "--model", "m", "hi"]),
        ("agy", ["agy", "-p", "--model", "m", "hi"]),
        ("ollama", ["ollama", "run", "m", "hi"]),
    ],
)
def test_each_provider_runs_its_own_cli_non_interactively(provider: str, expected: list[str]) -> None:
    calls: list[list[str]] = []
    assert is_provider(provider)
    agent = agent_for(provider, lambda command: calls.append(command) or "ok")
    assert agent.call("m", "hi") == "ok"
    assert calls == [expected]


def test_an_empty_prompt_never_reaches_a_paid_cli() -> None:
    calls: list[list[str]] = []
    agent = agent_for("claude", lambda command: calls.append(command) or "ok")
    with pytest.raises(ValueError, match="prompt"):
        agent.call("m", "  ")
    with pytest.raises(ValueError, match="model"):
        agent.call("", "hi")
    assert calls == []


def test_unknown_providers_are_rejected() -> None:
    assert not is_provider("gpt")


def test_a_failing_cli_raises_so_the_durable_step_retries() -> None:
    with pytest.raises(AgentError, match="exit code 3"):
        process_runner(["sh", "-c", "echo boom >&2; exit 3"])
    with pytest.raises(AgentError, match="Unable to start"):
        process_runner(["no-such-agent-cli-xyz"])


def test_the_cli_search_path_includes_user_install_dirs() -> None:
    path = command_path("claude")
    assert ".local/bin" in path
    assert "/opt/homebrew/bin" in path
