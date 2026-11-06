from pathlib import Path

import pytest

from agentsfile import AgentsfileError, Agentsfile, Transport, load, parse

FIXTURES = Path(__file__).resolve().parents[3] / "spec" / "fixtures"
INVALID = sorted((FIXTURES / "invalid").glob("*.agentsfile"))
EXPECT_PREFIX = "# expect: "


def _expected(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith(EXPECT_PREFIX)
    return first.removeprefix(EXPECT_PREFIX)


@pytest.fixture(scope="module")
def full() -> Agentsfile:
    return load(FIXTURES / "valid" / "full.agentsfile")


def test_every_invalid_fixture_is_collected() -> None:
    assert len(INVALID) == 30


@pytest.mark.parametrize("path", INVALID, ids=[path.stem for path in INVALID])
def test_invalid_fixture_reports_exact_error(path: Path) -> None:
    with pytest.raises(AgentsfileError) as caught:
        load(path)
    assert str(caught.value) == _expected(path)


def test_error_exposes_position_and_message() -> None:
    with pytest.raises(AgentsfileError) as caught:
        parse('version = "1"\nname = "x"\n')
    assert (caught.value.line, caught.value.col, caught.value.message) == (2, 1, 'unknown attribute "name" in file')


def test_minimal_has_version_and_no_profiles() -> None:
    parsed = load(str(FIXTURES / "valid" / "minimal.agentsfile"))
    assert parsed == Agentsfile(version="1", profiles=())


def test_profiles_keep_file_order(full: Agentsfile) -> None:
    assert full.version == "1"
    assert [profile.name for profile in full.profiles] == ["dev", "ci"]


def test_dev_profile_items(full: Agentsfile) -> None:
    dev = full.profiles[0]
    assert dev.description == "Everyday coding setup"
    assert [skill.name for skill in dev.skills] == ["code-review", "release-notes"]
    assert dev.skills[0].source == "./skills/code-review"
    assert dev.skills[0].description == "Reviews the current diff"
    assert dev.skills[1].source == "https://github.com/acme/skills/release-notes"
    assert dev.skills[1].description is None
    assert [command.name for command in dev.commands] == ["ship"]
    assert dev.commands[0].source == "./commands/ship.md"
    assert dev.commands[0].description == "Runs tests and opens a PR"


def test_agent_tester_with_trailing_comma_list(full: Agentsfile) -> None:
    tester = full.profiles[0].agents[0]
    assert tester.name == "tester"
    assert tester.source == "./agents/tester.md"
    assert tester.model == "claude-sonnet-5-5"
    assert tester.description is None
    assert tester.tools == ("Read", "Grep", "Bash")


def test_stdio_mcp_keeps_env_placeholder_literal(full: Agentsfile) -> None:
    github = full.profiles[0].mcps[0]
    assert github.name == "github"
    assert github.transport is Transport.STDIO
    assert github.command == "npx"
    assert github.args == ("-y", "@modelcontextprotocol/server-github")
    assert github.env == {"GITHUB_TOKEN": "${GITHUB_TOKEN}", "LOG_LEVEL": "info"}
    assert list(github.env) == ["GITHUB_TOKEN", "LOG_LEVEL"]
    assert github.url is None
    assert github.headers == {}


def test_ci_agent_decodes_escapes_and_defaults_tools(full: Agentsfile) -> None:
    ci = full.profiles[1]
    assert ci.description is None
    assert ci.skills == ()
    assert ci.commands == ()
    reviewer = ci.agents[0]
    assert reviewer.name == "reviewer"
    assert reviewer.source == "./agents/reviewer.md"
    assert reviewer.description == 'Says "no" to\tsloppy code'
    assert reviewer.model is None
    assert reviewer.tools == ()


def test_remote_mcps_use_http_and_sse(full: Agentsfile) -> None:
    docs, events = full.profiles[1].mcps
    assert docs.name == "docs"
    assert docs.transport is Transport.HTTP
    assert docs.url == "https://mcp.acme.dev/docs"
    assert docs.headers == {"Authorization": "Bearer ${DOCS_TOKEN}"}
    assert docs.command is None
    assert docs.args == ()
    assert docs.env == {}
    assert events.name == "events"
    assert events.transport is Transport.SSE
    assert events.url == "https://mcp.acme.dev/events"
    assert events.headers == {}
    assert events.description is None


def test_transports_in_file_order(full: Agentsfile) -> None:
    transports = [mcp.transport for profile in full.profiles for mcp in profile.mcps]
    assert transports == [Transport.STDIO, Transport.HTTP, Transport.SSE]


def test_columns_count_characters_not_bytes() -> None:
    with pytest.raises(AgentsfileError) as caught:
        parse('version = "ü" @')
    assert str(caught.value) == "line 1, col 15: unexpected character '@'"


def test_lexical_error_wins_over_syntax_error() -> None:
    with pytest.raises(AgentsfileError) as caught:
        parse('version {\n"open')
    assert str(caught.value) == "line 2, col 1: unterminated string"


def test_attribute_errors_follow_document_order() -> None:
    with pytest.raises(AgentsfileError) as caught:
        parse('version = "1"\nprofile "p" {\n  mcp "m" {\n    transport = "grpc"\n    env = "x"\n  }\n}\n')
    assert str(caught.value) == 'line 4, col 17: invalid transport "grpc", expected stdio, http or sse'


def test_missing_version_is_checked_after_profiles() -> None:
    with pytest.raises(AgentsfileError) as caught:
        parse('profile "p" {\n  skill "s" {\n  }\n}\n')
    assert str(caught.value) == 'line 2, col 3: missing required attribute "source" in skill "s"'


def test_same_name_allowed_across_kinds_and_profiles() -> None:
    parsed = parse(
        'version = "1"\n'
        'profile "a" { skill "x" { source = "s" } agent "x" { source = "g" } }\n'
        'profile "b" { skill "x" { source = "t" } }\n'
    )
    assert [p.skills[0].source for p in parsed.profiles] == ["s", "t"]
    assert parsed.profiles[0].agents[0].name == "x"
