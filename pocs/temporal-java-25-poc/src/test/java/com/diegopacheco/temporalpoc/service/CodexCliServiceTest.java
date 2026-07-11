package com.diegopacheco.temporalpoc.service;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CodexCliServiceTest {
    @Test
    void shouldTimeoutWithoutBlockingOnStdout() throws Exception {
        Path script = Files.createTempFile("fake-codex-", ".sh");
        Files.writeString(script, """
                #!/usr/bin/env bash
                printf 'started'
                sleep 5
                """);
        script.toFile().setExecutable(true);
        CodexCliService service = new CodexCliService(script.toString(), true, 1, "low");
        Instant start = Instant.now();
        assertThatThrownBy(() -> service.ask("research AAPL"))
                .isInstanceOf(CodexCliException.class)
                .hasMessageContaining("timed out");
        long elapsedMs = Duration.between(start, Instant.now()).toMillis();
        assertThat(elapsedMs).isLessThan(3000);
    }
}
