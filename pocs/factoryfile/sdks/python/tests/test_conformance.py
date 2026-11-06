from pathlib import Path

import pytest

from factoryfile import Agent, Factory, Factoryfile, FactoryfileError, Stage, load, parse

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


def test_every_invalid_fixture_is_collected() -> None:
    assert len(INVALID) == 63


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
