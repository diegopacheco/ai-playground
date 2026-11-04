import json
import tempfile
import unittest
from pathlib import Path

from cli.credentials import REFRESH_MARGIN_SECONDS, cached_token
from cli.main import parse_fields, parse_value
from cli.suite import load_suite
from common.http import HttpError, Request, Response, Router, json_response
from common.secrets import new_client, verify_secret
from logs.loki import parse_query, push_body, selector, validate, LogLine
from state.store import validate_key, validate_value


async def noop(_: Request) -> Response:
    return json_response(None)


class RouterTest(unittest.TestCase):
    def test_path_param_keeps_slashes_so_keys_can_be_namespaced(self) -> None:
        router = Router()
        router.add("GET", "/v1/state/{key:path}", noop)
        _, params = router.resolve("GET", "/v1/state/adr/use-rls")
        self.assertEqual(params, {"key": "adr/use-rls"})

    def test_wrong_method_is_405_and_unknown_path_is_404(self) -> None:
        router = Router()
        router.add("GET", "/v1/state", noop)
        with self.assertRaises(HttpError) as ctx:
            router.resolve("POST", "/v1/state")
        self.assertEqual(ctx.exception.status, 405)
        with self.assertRaises(HttpError) as ctx:
            router.resolve("GET", "/nope")
        self.assertEqual(ctx.exception.status, 404)


class ValidationTest(unittest.TestCase):
    def test_keys_reject_traversal_like_and_empty_values(self) -> None:
        self.assertEqual(validate_key("flag/new-ui"), "flag/new-ui")
        for bad in ["", "a b", "k" * 201, "tenant'--", "%"]:
            with self.assertRaises(HttpError):
                validate_key(bad)

    def test_values_over_64kb_are_rejected(self) -> None:
        validate_value("x" * 60_000)
        with self.assertRaises(HttpError) as ctx:
            validate_value("x" * 70_000)
        self.assertEqual(ctx.exception.status, 413)

    def test_client_secret_is_verified_against_the_hash_only(self) -> None:
        client, secret = new_client("c", ["state:read"])
        self.assertNotIn(secret, client.secret_hash)
        self.assertTrue(verify_secret(client, secret))
        self.assertFalse(verify_secret(client, secret + "x"))


class LokiTest(unittest.TestCase):
    def test_tenant_is_never_a_label_because_isolation_comes_from_the_org_header(self) -> None:
        body = push_body(LogLine(1, "info", "m", {"a": 1}))
        self.assertNotIn("tenant", body["streams"][0]["stream"])

    def test_level_filter_cannot_inject_logql(self) -> None:
        self.assertEqual(selector("warn"), '{service_name="agentinfra", level="warn"}')
        with self.assertRaises(HttpError):
            selector('info"} or {x="')

    def test_invalid_log_input_is_rejected(self) -> None:
        with self.assertRaises(HttpError):
            validate("fatal", "m", {})
        with self.assertRaises(HttpError):
            validate("info", "", {})
        with self.assertRaises(HttpError):
            validate("info", "m", [])

    def test_query_results_are_newest_first_across_streams(self) -> None:
        result = {
            "data": {
                "result": [
                    {"stream": {"level": "info"}, "values": [["1", json.dumps({"message": "old", "fields": {}})]]},
                    {"stream": {"level": "warn"}, "values": [["3", json.dumps({"message": "new", "fields": {}})]]},
                ]
            }
        }
        self.assertEqual([line.message for line in parse_query(result)], ["new", "old"])


class CliTest(unittest.TestCase):
    def test_values_are_json_when_possible_and_strings_otherwise(self) -> None:
        self.assertEqual(parse_value('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_value("plain text"), "plain text")
        self.assertEqual(parse_fields(["n=1", "s=x"]), {"n": 1, "s": "x"})
        with self.assertRaises(ValueError):
            parse_fields(["novalue"])

    def test_cached_token_is_refreshed_before_it_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secrets_dir = Path(tmp)
            (secrets_dir / "tokens").mkdir()
            (secrets_dir / "tokens" / "c.json").write_text(json.dumps({"access_token": "t", "expires_at": 1000}))
            self.assertEqual(cached_token(secrets_dir, "c", 1000 - REFRESH_MARGIN_SECONDS - 1), "t")
            self.assertIsNone(cached_token(secrets_dir, "c", 1000 - REFRESH_MARGIN_SECONDS))

    def test_suite_runs_commands_in_the_suite_folder_and_measures_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "answer.txt").write_text("from file")
            (folder / "suite.json").write_text(
                json.dumps(
                    {
                        "suite": "s",
                        "cases": [
                            {"name": "cmd", "command": ["cat", "answer.txt"], "checks": [{"type": "json_valid"}]},
                            {"name": "file", "output_file": "answer.txt", "checks": [{"type": "json_valid"}]},
                            {"name": "inline", "output": "given", "checks": [{"type": "json_valid"}]},
                        ],
                    }
                )
            )
            run = load_suite(folder / "suite.json")
        self.assertEqual([c["output"] for c in run.cases], ["from file", "from file", "given"])
        self.assertGreaterEqual(run.cases[0]["latency_ms"], 0)
        self.assertEqual(run.cases[0]["input"], "cat answer.txt")


if __name__ == "__main__":
    unittest.main()
