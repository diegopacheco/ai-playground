from pathlib import Path

import pytest

from agentsfile import AgentsfileError, Agentsfile, SkillFile, Transport, load, parse

FIXTURES = Path(__file__).resolve().parents[3] / "spec" / "fixtures"
INVALID = sorted((FIXTURES / "invalid").glob("*.agentsfile"))
EXPECT_PREFIX = "# expect: "


def _expected(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith(EXPECT_PREFIX)
    return first.removeprefix(EXPECT_PREFIX)


def _error(text: str) -> str:
    with pytest.raises(AgentsfileError) as caught:
        parse(text)
    return str(caught.value)


@pytest.fixture(scope="module")
def full() -> Agentsfile:
    return load(FIXTURES / "valid" / "full.agentsfile")


def test_every_invalid_fixture_is_collected() -> None:
    assert len(INVALID) == 51


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
    assert load(str(FIXTURES / "valid" / "minimal.agentsfile")) == Agentsfile(version="1", profiles=())


def test_full_keeps_profile_and_item_order(full: Agentsfile) -> None:
    dev, ci = full.profiles
    assert full.version == "1"
    assert [profile.name for profile in full.profiles] == ["dev", "ci"]
    assert [skill.name for skill in dev.skills] == ["code-review", "release-notes"]
    assert [mcp.name for mcp in ci.mcps] == ["docs", "events", "shared"]
    assert [plugin.name for plugin in ci.plugins] == ["toolkit", "security"]
    assert dev.description == "Everyday coding setup"
    assert dev.plugins == ()


def test_inline_skill_carries_its_whole_body_and_extra_files(full: Agentsfile) -> None:
    review, notes = full.profiles[0].skills
    assert review.source is None
    assert review.description == "Reviews the current diff"
    assert review.content == "# Code review\nRead the diff and list bugs first.\n\nThen list style issues."
    assert review.files == (
        SkillFile("scripts/diff.sh", None, "#!/usr/bin/env bash\ngit diff --stat"),
        SkillFile("checklist.md", "./skills/code-review/checklist.md", None),
    )
    assert notes.source == "https://github.com/acme/skills/release-notes"
    assert notes.content is None
    assert notes.files == ()


def test_inline_agent_and_command_keep_their_prompts(full: Agentsfile) -> None:
    tester = full.profiles[0].agents[0]
    assert tester.name == "tester"
    assert tester.description == "Writes and runs tests"
    assert tester.model == "claude-sonnet-5-5"
    assert tester.tools == ("Read", "Grep", "Bash")
    assert tester.content == "You write tests first.\n  Keep each test small."
    assert tester.source is None
    ship = full.profiles[0].commands[0]
    assert ship.name == "ship"
    assert ship.description == "Runs tests and opens a PR"
    assert ship.argument_hint == "<branch>"
    assert ship.model == "claude-haiku-5-5"
    assert ship.tools == ("Bash",)
    assert ship.content == "Run the tests, then open a PR for $ARGUMENTS."
    assert ship.source is None
    fix = full.profiles[1].commands[0]
    assert fix.source == "./commands/fix.md"
    assert fix.content is None
    assert fix.tools == ()
    assert fix.argument_hint is None
    assert fix.model is None


def test_mcp_servers_are_inline_or_referenced(full: Agentsfile) -> None:
    github = full.profiles[0].mcps[0]
    assert github.transport is Transport.STDIO
    assert github.command == "npx"
    assert github.args == ("-y", "@modelcontextprotocol/server-github")
    assert github.env == {"GITHUB_TOKEN": "${GITHUB_TOKEN}", "LOG_LEVEL": "info"}
    assert list(github.env) == ["GITHUB_TOKEN", "LOG_LEVEL"]
    assert github.url is None
    assert github.headers == {}
    assert github.source is None
    docs, events, shared = full.profiles[1].mcps
    assert docs.transport is Transport.HTTP
    assert docs.url == "https://mcp.acme.dev/docs"
    assert docs.headers == {"Authorization": "Bearer ${DOCS_TOKEN}"}
    assert docs.command is None
    assert docs.args == ()
    assert docs.env == {}
    assert events.transport is Transport.SSE
    assert events.url == "https://mcp.acme.dev/events"
    assert events.headers == {}
    assert shared.source == "./mcp/shared.json"
    assert shared.transport is None
    assert (shared.args, shared.env, shared.headers) == ((), {}, {})


def test_plugins_are_inline_bundles_or_referenced(full: Agentsfile) -> None:
    toolkit, security = full.profiles[1].plugins
    assert toolkit.version == "1.2.0"
    assert toolkit.author == "Platform team"
    assert toolkit.description == "Shared team tooling"
    assert toolkit.source is None
    assert toolkit.skills[0].name == "lint"
    assert toolkit.skills[0].content == "Run the linter and fix every error."
    assert toolkit.agents[0].name == "fixer"
    assert toolkit.agents[0].source == "./agents/fixer.md"
    assert toolkit.commands[0].name == "fix"
    assert toolkit.mcps[0].name == "search"
    assert toolkit.mcps[0].transport is Transport.HTTP
    assert toolkit.mcps[0].url == "https://mcp.acme.dev/search"
    assert security.source == "https://github.com/acme/plugins/security"
    assert (security.skills, security.agents, security.commands, security.mcps) == ((), (), (), ())


def test_quoted_strings_decode_escapes_and_optional_fields_default(full: Agentsfile) -> None:
    ci = full.profiles[1]
    assert ci.description is None
    assert ci.skills == ()
    reviewer = ci.agents[0]
    assert reviewer.source == "./agents/reviewer.md"
    assert reviewer.description == 'Says "no" to\tsloppy code'
    assert reviewer.tools == ()
    assert reviewer.model is None
    assert reviewer.content is None


def test_heredoc_edge_cases() -> None:
    parsed = load(FIXTURES / "valid" / "heredoc.agentsfile")
    assert parsed.version == "1"
    edge = parsed.profiles[0]
    assert edge.name == "edge"
    assert edge.description == ""
    assert edge.skills[0].name == "tabs"
    assert edge.skills[0].content == "first\n  second\n"
    assert edge.skills[1].name == "raw"
    assert edge.skills[1].content == '"quoted" \\n ${HOME} # not a comment'


def test_heredoc_strips_trailing_carriage_returns() -> None:
    parsed = parse('version = "1"\nprofile "p" {\n  skill "s" {\n    content = <<E\r\n    a\r\n      b\r\n    E\r\n  }\n}\n')
    assert parsed.profiles[0].skills[0].content == "a\n  b"


def test_positions_after_a_heredoc_follow_its_closing_line() -> None:
    assert _error("version = <<V\n1\nV\n  @") == "line 4, col 3: unexpected character '@'"


def test_heredoc_marker_errors_point_at_the_angles() -> None:
    assert _error("version = <<1V\n1\n1V\n") == "line 1, col 11: invalid heredoc marker"
    assert _error("version = <<V x\n1\nV\n") == "line 1, col 11: invalid heredoc marker"
    assert _error("version = <<V") == "line 1, col 11: unterminated heredoc"
    assert _error("version = <<V\n1\n") == "line 1, col 11: unterminated heredoc"


def test_columns_count_characters_not_bytes() -> None:
    assert _error('version = "1"\nprofile "é" { @ }') == "line 2, col 15: unexpected character '@'"


def test_lexical_error_wins_over_syntax_error() -> None:
    assert _error('version {\n"open') == "line 2, col 1: unterminated string"


def test_attribute_value_is_checked_before_later_statements() -> None:
    text = 'version = "1"\nprofile "p" {\n  mcp "m" {\n    transport = "grpc"\n    env = "x"\n  }\n}\n'
    assert _error(text) == 'line 4, col 17: invalid transport "grpc", expected stdio, http or sse'


def test_requires_is_checked_before_transport_rule() -> None:
    text = 'version = "1"\nprofile "p" {\n  mcp "m" {\n    source = "s"\n    transport = "stdio"\n  }\n}\n'
    assert _error(text) == 'line 3, col 3: mcp "m" cannot set both "source" and "transport"'


def test_plugin_version_is_not_the_file_version() -> None:
    parsed = parse('version = "1"\nprofile "p" {\n  plugin "x" {\n    version = "9"\n    source = "s"\n  }\n}\n')
    assert parsed.profiles[0].plugins[0].version == "9"


def test_missing_version_is_checked_after_profiles() -> None:
    assert _error('profile "p" {\n  skill "s" {\n  }\n}\n') == 'line 2, col 3: skill "s" requires "source" or "content"'


def test_same_name_allowed_across_kinds_and_profiles() -> None:
    parsed = parse(
        'version = "1"\n'
        'profile "a" { skill "x" { source = "s" } agent "x" { source = "g" } plugin "x" { source = "c" } }\n'
        'profile "b" { skill "x" { source = "t" } }\n'
    )
    assert [p.skills[0].source for p in parsed.profiles] == ["s", "t"]
    assert parsed.profiles[0].agents[0].name == "x"
    assert parsed.profiles[0].plugins[0].source == "c"
