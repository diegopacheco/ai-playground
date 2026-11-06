package io.agentsfile;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ValidFixturesTest {
    private static Agentsfile full;
    private static Profile dev;
    private static Profile ci;

    @BeforeAll
    static void load() {
        full = Agentsfile.load(Fixtures.valid("full.agentsfile"));
        dev = full.profiles().get(0);
        ci = full.profiles().get(1);
    }

    @Test
    void minimalFileHasVersionAndNoProfiles() {
        Agentsfile minimal = Agentsfile.load(Fixtures.valid("minimal.agentsfile"));
        assertEquals("1", minimal.version());
        assertEquals(List.of(), minimal.profiles());
    }

    @Test
    void profilesAndItemsKeepFileOrder() {
        assertEquals("1", full.version());
        assertEquals(List.of("dev", "ci"), full.profiles().stream().map(Profile::name).toList());
        assertEquals(List.of("code-review", "release-notes"), dev.skills().stream().map(Skill::name).toList());
        assertEquals(List.of("docs", "events", "shared"), ci.mcps().stream().map(Mcp::name).toList());
        assertEquals(List.of("toolkit", "security"), ci.plugins().stream().map(Plugin::name).toList());
        assertEquals(Optional.of("Everyday coding setup"), dev.description());
        assertEquals(Optional.empty(), ci.description());
        assertEquals(List.of(), dev.plugins());
    }

    @Test
    void inlineSkillCarriesItsWholeBodyAndExtraFiles() {
        assertEquals(new Skill("code-review", Optional.of("Reviews the current diff"), Optional.empty(),
                Optional.of("# Code review\nRead the diff and list bugs first.\n\nThen list style issues."),
                List.of(
                        new SkillFile("scripts/diff.sh", Optional.empty(),
                                Optional.of("#!/usr/bin/env bash\ngit diff --stat")),
                        new SkillFile("checklist.md", Optional.of("./skills/code-review/checklist.md"),
                                Optional.empty()))),
                dev.skills().get(0));
        assertEquals(new Skill("release-notes", Optional.empty(),
                        Optional.of("https://github.com/acme/skills/release-notes"), Optional.empty(), List.of()),
                dev.skills().get(1));
    }

    @Test
    void inlineAgentKeepsRelativeIndentationOfItsPrompt() {
        assertEquals(List.of(new Agent("tester", Optional.of("Writes and runs tests"),
                Optional.of("claude-sonnet-5-5"), List.of("Read", "Grep", "Bash"), Optional.empty(),
                Optional.of("You write tests first.\n  Keep each test small."))), dev.agents());
    }

    @Test
    void referencedAgentDecodesEscapesAndDefaultsOptionalFields() {
        assertEquals(List.of(new Agent("reviewer", Optional.of("Says \"no\" to\tsloppy code"), Optional.empty(),
                List.of(), Optional.of("./agents/reviewer.md"), Optional.empty())), ci.agents());
    }

    @Test
    void commandsAreInlineOrReferenced() {
        assertEquals(List.of(new Command("ship", Optional.of("Runs tests and opens a PR"), Optional.of("<branch>"),
                Optional.of("claude-haiku-5-5"), List.of("Bash"), Optional.empty(),
                Optional.of("Run the tests, then open a PR for $ARGUMENTS."))), dev.commands());
        assertEquals(List.of(new Command("fix", Optional.empty(), Optional.empty(), Optional.empty(), List.of(),
                Optional.of("./commands/fix.md"), Optional.empty())), ci.commands());
    }

    @Test
    void stdioMcpKeepsEnvOrderAndLeavesPlaceholdersLiteral() {
        Mcp github = dev.mcps().getFirst();
        assertEquals("github", github.name());
        assertEquals(Optional.of(Transport.STDIO), github.transport());
        assertEquals(Optional.empty(), github.source());
        assertEquals(Optional.of("npx"), github.command());
        assertEquals(List.of("-y", "@modelcontextprotocol/server-github"), github.args());
        assertEquals(Map.of("GITHUB_TOKEN", "${GITHUB_TOKEN}", "LOG_LEVEL", "info"), github.env());
        assertEquals(List.of("GITHUB_TOKEN", "LOG_LEVEL"), List.copyOf(github.env().keySet()));
        assertEquals(Optional.empty(), github.url());
        assertEquals(Map.of(), github.headers());
    }

    @Test
    void httpSseAndReferencedMcpsDefaultTheirCollections() {
        Mcp docs = ci.mcps().get(0);
        assertEquals(Optional.of(Transport.HTTP), docs.transport());
        assertEquals(Optional.of("https://mcp.acme.dev/docs"), docs.url());
        assertEquals(Map.of("Authorization", "Bearer ${DOCS_TOKEN}"), docs.headers());
        assertEquals(Optional.empty(), docs.command());
        assertEquals(List.of(), docs.args());
        assertEquals(Map.of(), docs.env());

        Mcp events = ci.mcps().get(1);
        assertEquals(Optional.of(Transport.SSE), events.transport());
        assertEquals(Optional.of("https://mcp.acme.dev/events"), events.url());
        assertEquals(Map.of(), events.headers());

        assertEquals(new Mcp("shared", Optional.empty(), Optional.of("./mcp/shared.json"), Optional.empty(),
                Optional.empty(), List.of(), Map.of(), Optional.empty(), Map.of()), ci.mcps().get(2));
    }

    @Test
    void pluginsAreInlineBundlesOrReferenced() {
        Plugin toolkit = ci.plugins().get(0);
        assertEquals(Optional.of("1.2.0"), toolkit.version());
        assertEquals(Optional.of("Platform team"), toolkit.author());
        assertEquals(Optional.of("Shared team tooling"), toolkit.description());
        assertEquals(Optional.empty(), toolkit.source());
        assertEquals(List.of(new Skill("lint", Optional.empty(), Optional.empty(),
                Optional.of("Run the linter and fix every error."), List.of())), toolkit.skills());
        assertEquals(Optional.of("./agents/fixer.md"), toolkit.agents().getFirst().source());
        assertEquals("fix", toolkit.commands().getFirst().name());
        assertEquals(Optional.of("https://mcp.acme.dev/search"), toolkit.mcps().getFirst().url());
        assertEquals(Optional.of(Transport.HTTP), toolkit.mcps().getFirst().transport());

        assertEquals(new Plugin("security", Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of("https://github.com/acme/plugins/security"), List.of(), List.of(), List.of(), List.of()),
                ci.plugins().get(1));
    }

    @Test
    void heredocEdgeCases() {
        Agentsfile file = Agentsfile.load(Fixtures.valid("heredoc.agentsfile"));
        assertEquals("1", file.version());
        Profile edge = file.profiles().getFirst();
        assertEquals("edge", edge.name());
        assertEquals(Optional.of(""), edge.description());
        assertEquals(Optional.of("first\n  second\n"), edge.skills().get(0).content());
        assertEquals(Optional.of("\"quoted\" \\n ${HOME} # not a comment"), edge.skills().get(1).content());
    }

    @Test
    void positionsAfterAHeredocFollowItsClosingLine() {
        AgentsfileException error = assertThrows(AgentsfileException.class,
                () -> Agentsfile.parse("version = <<V\n1\nV\n  @"));
        assertEquals("line 4, col 3: unexpected character '@'", error.getMessage());
    }

    @Test
    void sameNameInDifferentKindsIsAllowed() {
        Profile profile = Agentsfile.parse("version = \"1\"\nprofile \"p\" {\n skill \"x\" { source = \"a\" }\n"
                + " agent \"x\" { source = \"b\" }\n plugin \"x\" { source = \"c\" }\n}").profiles().getFirst();
        assertEquals("x", profile.skills().getFirst().name());
        assertEquals("x", profile.agents().getFirst().name());
        assertEquals("x", profile.plugins().getFirst().name());
    }

    @Test
    void modelCollectionsAreUnmodifiable() {
        Mcp github = dev.mcps().getFirst();
        assertThrows(UnsupportedOperationException.class, () -> github.env().put("X", "y"));
        assertThrows(UnsupportedOperationException.class, () -> github.args().add("x"));
        assertThrows(UnsupportedOperationException.class, () -> full.profiles().clear());
        assertThrows(UnsupportedOperationException.class, () -> dev.skills().getFirst().files().clear());
    }

    @Test
    void columnsCountCharactersNotBytes() {
        AgentsfileException error = assertThrows(AgentsfileException.class,
                () -> Agentsfile.parse("version = \"1\"\nprofile \"é\" { @ }"));
        assertEquals("line 2, col 15: unexpected character '@'", error.getMessage());
        assertEquals(2, error.line());
        assertEquals(15, error.col());
    }
}
