package com.github.diegopacheco.restate.agents;

import com.github.diegopacheco.restate.AgentException;
import com.github.diegopacheco.restate.config.AgentConfig;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentCallerTest {

    private final List<List<String>> calls = new ArrayList<>();
    private final AgentCaller caller = new AgentCaller(command -> {
        calls.add(command);
        return "  answer \n";
    });

    @Test
    void passesThePromptAsOneArgumentSoShellCharactersAreNeverInterpreted() {
        String answer = caller.ask(new AgentConfig(Provider.CLAUDE, "sonnet"), "a; rm -rf / $HOME");
        assertEquals("answer", answer);
        assertEquals(List.of("claude", "-p", "--model", "sonnet", "a; rm -rf / $HOME"), calls.getFirst());
    }

    @Test
    void eachProviderUsesItsOwnCli() {
        assertEquals("codex", Provider.CODEX.command("m", "p").getFirst());
        assertEquals(List.of("ollama", "run", "llama3.2", "p"), Provider.OLLAMA.command("llama3.2", "p"));
    }

    @Test
    void refusesABlankPromptWithoutSpendingAnAgentCall() {
        assertThrows(AgentException.class, () -> caller.ask(new AgentConfig(Provider.CLAUDE, "m"), "  "));
        assertTrue(calls.isEmpty());
    }

    @Test
    void searchesUserCliFoldersFirstBecauseTheServiceMayStartWithAMinimalPath() {
        String path = ProcessRunner.commandPath(Path.of("/home/u"), "/usr/bin:/usr/local/bin");
        assertTrue(path.startsWith("/home/u/.local/bin:/home/u/.bun/bin:/opt/homebrew/bin:/usr/local/bin"));
        assertEquals(1, path.split("/usr/local/bin", -1).length - 1);
    }

    @Test
    void reportsAFailingCliWithItsErrorSoTheRunShowsWhy() {
        AgentException error = assertThrows(AgentException.class, () -> new ProcessRunner().run(List.of("sh", "-c", "echo nope >&2; exit 3")));
        assertTrue(error.getMessage().contains("3") && error.getMessage().contains("nope"));
    }
}
