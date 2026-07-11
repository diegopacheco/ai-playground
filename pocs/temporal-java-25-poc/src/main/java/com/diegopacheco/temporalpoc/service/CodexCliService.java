package com.diegopacheco.temporalpoc.service;

import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.List;

@Service
public class CodexCliService {
    private static final Logger log = LoggerFactory.getLogger(CodexCliService.class);
    private final String command;
    private final boolean enabled;

    public CodexCliService(@Value("${codex.command}") String command, @Value("${codex.enabled}") boolean enabled) {
        this.command = command;
        this.enabled = enabled;
    }

    public String ask(String prompt) {
        Prompt aiPrompt = new Prompt(new UserMessage(prompt));
        String content = aiPrompt.getContents();
        log.info("codex cli request prepared enabled={} command={} promptLength={}", enabled, command, content.length());
        if (!enabled) {
            log.warn("codex cli disabled promptLength={}", content.length());
            return "Codex CLI disabled. " + content;
        }
        Instant start = Instant.now();
        try {
            log.info("codex cli process starting command={} timeoutMinutes=5", command);
            Process process = new ProcessBuilder(List.of(command, "exec", "--skip-git-repo-check", content))
                    .redirectErrorStream(true)
                    .start();
            boolean completed = process.waitFor(Duration.ofMinutes(5));
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            long elapsedMs = Duration.between(start, Instant.now()).toMillis();
            if (!completed) {
                process.destroyForcibly();
                log.error("codex cli timed out elapsedMs={} promptLength={}", elapsedMs, content.length());
                return "Codex CLI timed out.";
            }
            if (process.exitValue() != 0) {
                log.error("codex cli failed exitCode={} elapsedMs={} outputLength={}", process.exitValue(), elapsedMs, output.length());
                return "Codex CLI failed: " + output;
            }
            log.info("codex cli completed exitCode={} elapsedMs={} outputLength={}", process.exitValue(), elapsedMs, output.length());
            return output;
        } catch (Exception e) {
            log.error("codex cli unavailable error={}", e.getMessage(), e);
            return "Codex CLI unavailable: " + e.getMessage();
        }
    }
}
