from pathlib import Path

import pytest

from factoryfile import Agent, Exhausted, Factory, Factoryfile, FactoryfileError, Route, Stage, load, parse

FIXTURES = Path(__file__).resolve().parents[3] / "spec" / "fixtures"
INVALID = sorted((FIXTURES / "invalid").glob("*.factoryfile"))
EXPECT_PREFIX = "# expect: "


def _expected(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith(EXPECT_PREFIX)
    return first.removeprefix(EXPECT_PREFIX)


def _error(text: str) -> str:
    with pytest.raises(FactoryfileError) as caught:
        parse(text)
    return str(caught.value)


def _agent(factory: Factory, name: str) -> Agent:
    agent = factory.agent(name)
    assert agent is not None
    return agent


def _stage(factory: Factory, name: str) -> Stage:
    stage = factory.stage(name)
    assert stage is not None
    return stage


@pytest.fixture(scope="module")
def full() -> Factoryfile:
    return load(FIXTURES / "valid" / "full.factoryfile")


@pytest.fixture(scope="module")
def gate() -> Factory:
    return load(FIXTURES / "valid" / "routes.factoryfile").factories[0]


def test_every_invalid_fixture_is_collected() -> None:
    assert len(INVALID) == 82


@pytest.mark.parametrize("path", INVALID, ids=[path.stem for path in INVALID])
def test_invalid_fixture_reports_exact_error(path: Path) -> None:
    with pytest.raises(FactoryfileError) as caught:
        load(path)
    assert str(caught.value) == _expected(path)


def test_error_exposes_position_and_message() -> None:
    with pytest.raises(FactoryfileError) as caught:
        parse('version = "1"\nname = "x"\n')
    assert (caught.value.line, caught.value.col, caught.value.message) == (2, 1, 'unknown attribute "name" in file')


def test_minimal_file_has_no_factories() -> None:
    assert load(str(FIXTURES / "valid" / "minimal.factoryfile")) == Factoryfile(version="1", factories=())


def test_factories_and_items_keep_document_order(full: Factoryfile) -> None:
    scrum = full.factories[0]
    assert [factory.name for factory in full.factories] == ["scrum", "solo", "empty"]
    assert [stage.name for stage in scrum.stages] == ["refinement", "planning", "build", "review", "retro"]
    assert [loop.name for loop in scrum.loops] == ["sprint", "daily"]


def test_explicit_entry_wins_and_missing_entry_falls_back_to_the_first_stage(full: Factoryfile) -> None:
    scrum, solo, empty = full.factories
    assert scrum.entry == "refinement"
    assert solo.entry == "code"
    assert empty.entry is None


def test_inline_agent_carries_its_whole_prompt_and_source_agent_points_away(full: Factoryfile) -> None:
    scrum = full.factories[0]
    owner = _agent(scrum, "product-owner")
    assert owner.role == "product owner"
    assert owner.model == "claude-opus-5-5"
    assert owner.tools == ("Read", "Write")
    assert owner.source is None
    assert owner.content == "You keep the backlog ordered by value.\n\nEvery story has acceptance criteria."
    developer = _agent(scrum, "developer")
    assert developer.source == "./agents/developer.md"
    assert developer.content is None
    assert developer.model is None


def test_stages_hold_the_graph_edges_and_their_artifacts(full: Factoryfile) -> None:
    scrum = full.factories[0]
    build = _stage(scrum, "build")
    assert build.agents == ("developer",)
    assert build.next == ("review", "retro")
    assert build.artifacts == ("src/", "tests/")
    assert _stage(scrum, "review").next == (), "a stage without next ends the factory"
    assert _stage(scrum, "retro").agents == ()
    assert _stage(scrum, "planning").content == "Pick the stories for the sprint goal."


def test_loops_are_back_edges_with_an_exit_condition_and_a_bound(full: Factoryfile) -> None:
    sprint, daily = full.factories[0].loops
    assert (sprint.from_, sprint.to) == ("review", "planning")
    assert sprint.until == "the backlog is empty"
    assert sprint.max == 26
    assert (daily.from_, daily.to) == ("build", "build")
    assert (daily.until, daily.max) == (None, None)


def test_heredocs_strip_shared_indent_and_keep_text_raw() -> None:
    file = load(FIXTURES / "valid" / "heredoc.factoryfile")
    edge = file.factories[0]
    assert file.version == "1"
    assert edge.description == ""
    assert edge.agents[0].content == "first\n  second\n"
    assert edge.agents[1].content == '"quoted" \\n ${HOME} # not a comment'
    assert edge.stages[0].agents == ("tabs", "raw")


def test_columns_count_characters_not_bytes() -> None:
    error = _error('version = "1"\nfactory "é" { owner = "x" }\n')
    assert error == 'line 2, col 15: unknown attribute "owner" in factory'


def test_a_cycle_through_next_is_legal_for_a_parser() -> None:
    text = (
        'version = "1"\nfactory "x" {\n'
        '  stage "a" { agents = [] next = ["b"] }\n'
        '  stage "b" { agents = [] next = ["a"] }\n}\n'
    )
    assert parse(text).factories[0].stages[1].next == ("a",)


def test_v11_factory_carries_title_flow_backlog_and_agent_providers(gate: Factory, full: Factoryfile) -> None:
    assert (gate.title, gate.flow, gate.backlog) == ("Approval Gate", "pull", "optional")
    assert _agent(gate, "maker").provider == "codex"
    assert _agent(gate, "approver").provider == "claude"
    assert full.factories[0].title is None, "a v1 factory has no title"


def test_v11_command_stages_need_no_agents_and_keep_their_run_and_end(gate: Factory, full: Factoryfile) -> None:
    test = _stage(gate, "test")
    assert (test.run, test.agents, test.label) == ("test", (), "Test")
    assert _stage(gate, "ship").script == "scripts/ship.sh"
    done = _stage(gate, "done")
    assert (done.end, done.at) == ("succeeded", (1, 1))
    assert _stage(full.factories[0], "build").at == (), "a stage without at"


def test_v11_agent_stages_carry_their_prompt_file_skills_output_check_and_verdict(gate: Factory) -> None:
    implement = _stage(gate, "implement")
    assert implement.source == "prompts/implement.md"
    assert implement.skills == ("skills/tdd.md", "skills/git.md")
    assert implement.visits == 4
    approve = _stage(gate, "approve")
    assert (approve.output, approve.check) == ("approval.json", "verdict")
    assert approve.verdict == {
        "field": "approved",
        "yes": "approved",
        "no": "rejected",
        "list": "reasons",
        "notes": "notes",
    }
    build = _stage(gate, "build")
    assert (build.tasks, build.parallel, build.verdict) == ("split", 10, None)


def test_v11_routes_branch_by_outcome_and_fall_back_when_their_target_is_exhausted(gate: Factory) -> None:
    assert _stage(gate, "test").routes == (
        Route("passed", "approve", label="pass"),
        Route("failed", "implement", label="fail", reason="the tests failed"),
    )
    assert _stage(gate, "approve").routes[1] == Route("rejected", "implement", cycle=True)
    assert _stage(gate, "implement").routes == (
        Route(
            "next",
            "test",
            exhausted=Exhausted(
                "approve",
                label="first fallback",
                reason="test already ran as often as allowed",
                exhausted=Exhausted("done", label="last fallback"),
            ),
        ),
    )
    assert _stage(gate, "split").routes == (), "a stage with only next has no routes"
