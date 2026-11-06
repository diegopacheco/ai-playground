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
    void profilesKeepFileOrder() {
        assertEquals("1", full.version());
        assertEquals(List.of("dev", "ci"), full.profiles().stream().map(Profile::name).toList());
        assertEquals(Optional.of("Everyday coding setup"), dev.description());
        assertEquals(Optional.empty(), ci.description());
    }

    @Test
    void devSkillsKeepOrderAndOptionalDescription() {
        assertEquals(List.of(
                new Skill("code-review", "./skills/code-review", Optional.of("Reviews the current diff")),
                new Skill("release-notes", "https://github.com/acme/skills/release-notes", Optional.empty())
        ), dev.skills());
        assertEquals(List.of(), ci.skills());
    }

    @Test
    void agentToolsAcceptTrailingCommaAndDefaultToEmpty() {
        assertEquals(List.of(new Agent("tester", "./agents/tester.md", Optional.empty(),
                Optional.of("claude-sonnet-5-5"), List.of("Read", "Grep", "Bash"))), dev.agents());
        Agent reviewer = ci.agents().getFirst();
        assertEquals(List.of(), reviewer.tools());
        assertEquals(Optional.empty(), reviewer.model());
    }

    @Test
    void stringEscapesAreDecoded() {
        Agent reviewer = ci.agents().getFirst();
        assertEquals("reviewer", reviewer.name());
        assertEquals("./agents/reviewer.md", reviewer.source());
        assertEquals(Optional.of("Says \"no\" to\tsloppy code"), reviewer.description());
    }

    @Test
    void commandsAreMapped() {
        assertEquals(List.of(new Command("ship", "./commands/ship.md", Optional.of("Runs tests and opens a PR"))),
                dev.commands());
        assertEquals(List.of(), ci.commands());
    }

    @Test
    void stdioMcpKeepsEnvOrderAndLeavesPlaceholdersLiteral() {
        Mcp github = dev.mcps().getFirst();
        assertEquals("github", github.name());
        assertEquals(Transport.STDIO, github.transport());
        assertEquals(Optional.of("npx"), github.command());
        assertEquals(List.of("-y", "@modelcontextprotocol/server-github"), github.args());
        assertEquals(Map.of("GITHUB_TOKEN", "${GITHUB_TOKEN}", "LOG_LEVEL", "info"), github.env());
        assertEquals(List.of("GITHUB_TOKEN", "LOG_LEVEL"), List.copyOf(github.env().keySet()));
        assertEquals(Optional.empty(), github.url());
        assertEquals(Map.of(), github.headers());
        assertEquals(Optional.empty(), github.description());
    }

    @Test
    void httpAndSseMcpsDefaultArgsEnvAndHeaders() {
        assertEquals(List.of("docs", "events"), ci.mcps().stream().map(Mcp::name).toList());
        Mcp docs = ci.mcps().get(0);
        assertEquals(Transport.HTTP, docs.transport());
        assertEquals(Optional.of("https://mcp.acme.dev/docs"), docs.url());
        assertEquals(Map.of("Authorization", "Bearer ${DOCS_TOKEN}"), docs.headers());
        assertEquals(Optional.empty(), docs.command());
        assertEquals(List.of(), docs.args());
        assertEquals(Map.of(), docs.env());

        Mcp events = ci.mcps().get(1);
        assertEquals(Transport.SSE, events.transport());
        assertEquals(Optional.of("https://mcp.acme.dev/events"), events.url());
        assertEquals(Map.of(), events.headers());
        assertEquals(Map.of(), events.env());
        assertEquals(List.of(), events.args());
    }

    @Test
    void modelCollectionsAreUnmodifiable() {
        Mcp github = dev.mcps().getFirst();
        assertThrows(UnsupportedOperationException.class, () -> github.env().put("X", "y"));
        assertThrows(UnsupportedOperationException.class, () -> github.args().add("x"));
        assertThrows(UnsupportedOperationException.class, () -> full.profiles().clear());
    }

    @Test
    void columnsCountCharactersNotBytes() {
        AgentsfileException error = assertThrows(AgentsfileException.class,
                () -> Agentsfile.parse("version = \"éé\" @"));
        assertEquals("line 1, col 16: unexpected character '@'", error.getMessage());
        assertEquals(1, error.line());
        assertEquals(16, error.col());
    }
}
