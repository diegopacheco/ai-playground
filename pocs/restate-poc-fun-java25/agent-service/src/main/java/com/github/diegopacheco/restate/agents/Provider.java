package com.github.diegopacheco.restate.agents;

import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

public enum Provider {
    CLAUDE("claude"),
    CODEX("codex"),
    AGY("agy"),
    OLLAMA("ollama");

    private final String cli;

    Provider(String cli) {
        this.cli = cli;
    }

    @JsonValue
    public String cli() {
        return cli;
    }

    public static Optional<Provider> parse(String value) {
        return Arrays.stream(values()).filter(provider -> provider.cli.equals(value)).findFirst();
    }

    public List<String> command(String model, String prompt) {
        return switch (this) {
            case CLAUDE -> List.of("claude", "-p", "--model", model, prompt);
            case CODEX -> List.of("codex", "exec", "--model", model, prompt);
            case AGY -> List.of("agy", "-p", "--model", model, prompt);
            case OLLAMA -> List.of("ollama", "run", model, prompt);
        };
    }
}
