import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_sdk import (
    AgyAgent,
    ClaudeCodeAgent,
    CodexAgent,
    ModelQuota,
    OllamaAgent,
    Usage,
    latest_rate_limits,
    parse_agy_usage,
    parse_claude_usage,
    parse_codex_usage,
)


class AgentTest(unittest.TestCase):
    def test_providers_build_argument_arrays_without_a_shell(self):
        commands = []

        def runner(command):
            commands.append(command)
            return "ok"

        agents = [ClaudeCodeAgent(runner), CodexAgent(runner), AgyAgent(runner), OllamaAgent(runner)]
        for agent in agents:
            self.assertEqual("ok", agent.call("model-a", "hello world", ["--flag"]))
        self.assertEqual(["claude", "-p", "--model", "model-a", "--flag", "hello world"], commands[0])
        self.assertEqual(["ollama", "run", "--flag", "model-a", "hello world"], commands[3])
        self.assertEqual(["agy", "--model", "model-a", "--flag", "-p", "hello world"], commands[2])

    def test_required_inputs_are_explicit(self):
        agent = OllamaAgent(lambda command: "unexpected")
        with self.assertRaisesRegex(ValueError, "model is required"):
            agent.call("", "hello")
        with self.assertRaisesRegex(ValueError, "prompt is required"):
            agent.call("model", "")

    def test_ollama_is_always_available_because_it_runs_locally(self):
        usage = OllamaAgent(lambda command: "unexpected").usage()
        self.assertEqual("100%", usage.current_limit)
        self.assertEqual("100%", usage.weekly_limit)
        self.assertEqual("never", usage.time_to_reset_session)
        self.assertEqual("never", usage.time_to_reset_week)

    def test_agy_reports_a_quota_for_every_model(self):
        commands = []
        payload = json.dumps({
            "models": [
                {"label": "Gemini 3 Pro", "remainingPercentage": 0.99, "timeUntilResetMs": 3000000},
                {"label": "Claude Opus 4.6", "remainingPercentage": 0.42, "timeUntilResetMs": 90000},
            ]
        })

        def runner(command):
            commands.append(command)
            return payload

        usage = AgyAgent(runner).usage()
        self.assertEqual(["antigravity-usage", "quota", "--json"], commands[0])
        self.assertEqual("42%", usage.current_limit)
        self.assertEqual("1m", usage.time_to_reset_session)
        self.assertEqual("unavailable", usage.weekly_limit)
        self.assertEqual(
            [ModelQuota("Gemini 3 Pro", "99%", "50m"), ModelQuota("Claude Opus 4.6", "42%", "1m")],
            list(usage.models),
        )

    def test_agy_ignores_autocomplete_only_models_because_they_hold_a_separate_quota(self):
        payload = {
            "models": [
                {"label": "Gemini 3 Pro", "remainingPercentage": 0.8, "timeUntilResetMs": 60000},
                {"label": "Gemini 2.5 Flash", "remainingPercentage": 0.05, "timeUntilResetMs": 60000, "isAutocompleteOnly": True},
            ]
        }
        usage = parse_agy_usage(payload)
        self.assertEqual([ModelQuota("Gemini 3 Pro", "80%", "1m")], list(usage.models))
        self.assertEqual("80%", usage.current_limit)

    def test_agy_keeps_the_most_constrained_quota_for_a_repeated_model_label(self):
        payload = {
            "models": [
                {"label": "Gemini 3.1 Pro (High)", "modelId": "gemini-pro-agent", "remainingPercentage": 0.9, "timeUntilResetMs": 60000},
                {"label": "Gemini 3.1 Pro (High)", "modelId": "gemini-3.1-pro-high", "remainingPercentage": 0.4, "timeUntilResetMs": 60000},
            ]
        }
        self.assertEqual([ModelQuota("Gemini 3.1 Pro (High)", "40%", "1m")], list(parse_agy_usage(payload).models))

    def test_agy_usage_survives_a_banner_printed_before_the_json(self):
        payload = '\nAntigravity Quota Status\n{"models":[{"label":"Gemini 3 Pro","remainingPercentage":0.25,"timeUntilResetMs":60000}]}\n'
        self.assertEqual("25%", AgyAgent(lambda command: payload).usage().current_limit)

    def test_agy_usage_is_unavailable_when_antigravity_usage_fails(self):
        def runner(command):
            raise RuntimeError("not installed")

        self.assertEqual(Usage(), AgyAgent(runner).usage())

    def test_claude_reads_both_oauth_windows_as_remaining_capacity(self):
        usage = parse_claude_usage({
            "five_hour": {"utilization": 23, "resets_at": "2126-08-17T05:00:00+00:00"},
            "seven_day": {"utilization": 41, "resets_at": "2126-08-23T05:00:00+00:00"},
        })
        self.assertEqual("77%", usage.current_limit)
        self.assertEqual("59%", usage.weekly_limit)
        self.assertNotEqual("unavailable", usage.time_to_reset_session)

    def test_claude_usage_asks_the_keychain_and_never_puts_the_token_in_the_arguments(self):
        commands = []

        def runner(command):
            commands.append(command)
            if command[0] == "security":
                return json.dumps({"claudeAiOauth": {"accessToken": "secret-token"}})
            config = Path(command[3]).read_text(encoding="utf-8")
            assert "secret-token" in config
            return json.dumps({"five_hour": {"utilization": 10, "resets_at": None}})

        environment = dict(os.environ)
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        try:
            usage = ClaudeCodeAgent(runner).usage()
        finally:
            os.environ.clear()
            os.environ.update(environment)
        self.assertEqual("90%", usage.current_limit)
        self.assertEqual(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"], commands[0])
        self.assertEqual(["curl", "-sS", "-K"], commands[1][:3])
        self.assertNotIn("secret-token", " ".join(commands[1]))
        self.assertFalse(Path(commands[1][3]).exists())

    def test_claude_usage_is_unavailable_without_a_token(self):
        def runner(command):
            raise RuntimeError("no keychain")

        with TemporaryDirectory() as directory:
            environment = dict(os.environ)
            os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            os.environ["CLAUDE_CONFIG_DIR"] = directory
            try:
                self.assertEqual(Usage(), ClaudeCodeAgent(runner).usage())
            finally:
                os.environ.clear()
                os.environ.update(environment)

    def test_codex_reports_only_the_windows_the_provider_returns(self):
        usage = parse_codex_usage({
            "primary": {"used_percent": 52, "window_minutes": 10080, "resets_at": 4102444800},
            "secondary": None,
        })
        self.assertEqual("48%", usage.weekly_limit)
        self.assertEqual("unavailable", usage.current_limit)
        self.assertEqual("unavailable", usage.time_to_reset_session)

    def test_codex_without_any_window_is_unavailable(self):
        self.assertEqual(Usage(), parse_codex_usage({"primary": None, "secondary": None}))

    def test_codex_reads_the_newest_session_that_carries_rate_limits(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            sessions = home / "sessions" / "2026"
            sessions.mkdir(parents=True)
            (sessions / "old.jsonl").write_text(
                json.dumps({"payload": {"rate_limits": {"primary": {"used_percent": 90, "window_minutes": 300, "resets_at": 4102444800}}}}) + "\n",
                encoding="utf-8",
            )
            (sessions / "empty.jsonl").write_text(json.dumps({"payload": {"rate_limits": {"primary": None, "secondary": None}}}) + "\n", encoding="utf-8")
            os.utime(sessions / "old.jsonl", (1, 1))
            limits = latest_rate_limits(home)
            self.assertEqual(90, limits["primary"]["used_percent"])
            environment = dict(os.environ)
            os.environ["CODEX_HOME"] = str(home)
            try:
                self.assertEqual("10%", CodexAgent(lambda command: "unexpected").usage().current_limit)
            finally:
                os.environ.clear()
                os.environ.update(environment)


if __name__ == "__main__":
    unittest.main()
