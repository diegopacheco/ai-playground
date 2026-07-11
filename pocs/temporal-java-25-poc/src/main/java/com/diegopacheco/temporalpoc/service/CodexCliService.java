package com.diegopacheco.temporalpoc.service;

import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.List;

@Service
public class CodexCliService {
    private static final Logger log = LoggerFactory.getLogger(CodexCliService.class);
    private final String command;
    private final boolean enabled;
    private final int timeoutSeconds;

    public CodexCliService(@Value("${codex.command}") String command, @Value("${codex.enabled}") boolean enabled, @Value("${codex.timeout-seconds}") int timeoutSeconds) {
        this.command = command;
        this.enabled = enabled;
        this.timeoutSeconds = timeoutSeconds;
    }

    public String ask(String prompt) {
        Prompt aiPrompt = new Prompt(new UserMessage(prompt));
        String content = aiPrompt.getContents();
        log.info("codex step=prepare enabled={} command={} timeoutSeconds={} promptLength={}", enabled, command, timeoutSeconds, content.length());
        if (!enabled) {
            log.warn("codex step=skip reason=disabled promptLength={}", content.length());
            return "Codex CLI disabled. " + content;
        }
        Instant start = Instant.now();
        Path output = null;
        try {
            output = Files.createTempFile("codex-research-", ".txt");
            log.info("codex step=start command={} timeoutSeconds={} outputFile={}", command, timeoutSeconds, output);
            Process process = new ProcessBuilder(List.of(
                    command,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--ignore-rules",
                    "--cd",
                    System.getProperty("java.io.tmpdir"),
                    "--sandbox",
                    "read-only",
                    "--color",
                    "never",
                    "--output-last-message",
                    output.toString(),
                    content
            ))
                    .redirectErrorStream(true)
                    .start();
            log.info("codex step=wait timeoutSeconds={}", timeoutSeconds);
            boolean completed = process.waitFor(Duration.ofSeconds(timeoutSeconds));
            long elapsedMs = Duration.between(start, Instant.now()).toMillis();
            if (!completed) {
                process.destroyForcibly();
                boolean killed = process.waitFor(Duration.ofSeconds(2));
                log.error("codex step=timeout elapsedMs={} timeoutSeconds={} promptLength={}", elapsedMs, timeoutSeconds, content.length());
                return "Codex CLI timed out. killed=" + killed;
            }
            String stdout = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            if (process.exitValue() != 0) {
                log.error("codex step=failed exitCode={} elapsedMs={} stdoutLength={}", process.exitValue(), elapsedMs, stdout.length());
                return "Codex CLI failed: " + stdout;
            }
            String result = Files.exists(output) ? Files.readString(output).trim() : stdout;
            if (result.isBlank()) {
                result = stdout;
            }
            log.info("codex step=responded exitCode={} elapsedMs={} stdoutLength={} resultLength={}", process.exitValue(), elapsedMs, stdout.length(), result.length());
            return result;
        } catch (Exception e) {
            log.error("codex step=unavailable error={}", e.getMessage(), e);
            return "Codex CLI unavailable: " + e.getMessage();
        } finally {
            if (output != null) {
                try {
                    Files.deleteIfExists(output);
                } catch (Exception e) {
                    log.warn("codex step=cleanup_failed file={} error={}", output, e.getMessage());
                }
            }
        }
    }
}
