package io.factoryfile;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ValidFixturesTest {
    private static Factoryfile full;
    private static Factory scrum;

    @BeforeAll
    static void load() {
        full = Factoryfile.load(Fixtures.valid("full.factoryfile"));
        scrum = full.factories().getFirst();
    }

    @Test
    void minimalFileHasNoFactories() {
        Factoryfile minimal = Factoryfile.load(Fixtures.valid("minimal.factoryfile"));
        assertEquals("1", minimal.version());
        assertEquals(List.of(), minimal.factories());
    }

    @Test
    void factoriesAndItemsKeepDocumentOrder() {
        assertEquals(List.of("scrum", "solo", "empty"), full.factories().stream().map(Factory::name).toList());
        assertEquals(List.of("refinement", "planning", "build", "review", "retro"),
                scrum.stages().stream().map(Stage::name).toList());
        assertEquals(List.of("sprint", "daily"), scrum.loops().stream().map(Loop::name).toList());
    }

    @Test
    void explicitEntryWinsAndMissingEntryFallsBackToTheFirstStage() {
        assertEquals(Optional.of("refinement"), full.factories().get(0).entry());
        assertEquals(Optional.of("code"), full.factories().get(1).entry());
        assertEquals(Optional.empty(), full.factories().get(2).entry());
    }

    @Test
    void inlineAgentCarriesItsWholePromptAndSourceAgentPointsAway() {
        Agent po = scrum.agent("product-owner").orElseThrow();
        assertEquals(Optional.of("product owner"), po.role());
        assertEquals(Optional.of("claude-opus-5-5"), po.model());
        assertEquals(List.of("Read", "Write"), po.tools());
        assertEquals(Optional.empty(), po.source());
        assertEquals(Optional.of("You keep the backlog ordered by value.\n\nEvery story has acceptance criteria."),
                po.content());
        Agent dev = scrum.agent("developer").orElseThrow();
        assertEquals(Optional.of("./agents/developer.md"), dev.source());
        assertEquals(Optional.empty(), dev.content());
        assertEquals(Optional.empty(), dev.model());
    }

    @Test
    void stagesHoldTheGraphEdgesAndTheirArtifacts() {
        Stage build = scrum.stage("build").orElseThrow();
        assertEquals(List.of("developer"), build.agents());
        assertEquals(List.of("review", "retro"), build.next());
        assertEquals(List.of("src/", "tests/"), build.artifacts());
        assertTrue(scrum.stage("review").orElseThrow().next().isEmpty(), "a stage without next ends the factory");
        assertTrue(scrum.stage("retro").orElseThrow().agents().isEmpty());
        assertEquals(Optional.of("Pick the stories for the sprint goal."),
                scrum.stage("planning").orElseThrow().content());
    }

    @Test
    void loopsAreBackEdgesWithAnExitConditionAndABound() {
        assertEquals(new Loop("sprint", Optional.of("One sprint per pass"), "review", "planning",
                Optional.of("the backlog is empty"), Optional.of(26)), scrum.loops().get(0));
        assertEquals(new Loop("daily", Optional.empty(), "build", "build", Optional.empty(), Optional.empty()),
                scrum.loops().get(1));
    }

    @Test
    void heredocsStripSharedIndentAndKeepTextRaw() {
        Factoryfile file = Factoryfile.load(Fixtures.valid("heredoc.factoryfile"));
        assertEquals("1", file.version());
        Factory edge = file.factories().getFirst();
        assertEquals(Optional.of(""), edge.description());
        assertEquals(Optional.of("first\n  second\n"), edge.agents().get(0).content());
        assertEquals(Optional.of("\"quoted\" \\n ${HOME} # not a comment"), edge.agents().get(1).content());
        assertEquals(List.of("tabs", "raw"), edge.stages().getFirst().agents());
    }

    @Test
    void columnsCountCharactersNotBytes() {
        FactoryfileException error = assertThrows(FactoryfileException.class,
                () -> Factoryfile.parse("version = \"1\"\nfactory \"é\" { owner = \"x\" }\n"));
        assertEquals("line 2, col 15: unknown attribute \"owner\" in factory", error.getMessage());
        assertEquals(2, error.line());
        assertEquals(15, error.col());
    }

    @Test
    void aCycleThroughNextIsLegalForAParser() {
        Factoryfile file = Factoryfile.parse("version = \"1\"\nfactory \"x\" {\n"
                + "  stage \"a\" { agents = [] next = [\"b\"] }\n"
                + "  stage \"b\" { agents = [] next = [\"a\"] }\n}\n");
        assertEquals(List.of("a"), file.factories().getFirst().stages().get(1).next());
    }

    @Test
    void modelCollectionsAreUnmodifiable() {
        assertThrows(UnsupportedOperationException.class, () -> full.factories().clear());
        assertThrows(UnsupportedOperationException.class, () -> scrum.stages().clear());
        assertThrows(UnsupportedOperationException.class, () -> scrum.stages().getFirst().next().add("x"));
    }
}
