package com.github.diegopacheco.devadminconsole.ai;

import java.util.Arrays;
import java.util.List;

public enum AgentCli {
    CLAUDE("claude", List.of("-p"), "claude -p", "sonnet"),
    CODEX("codex", List.of("exec"), "codex exec", "gpt-5-codex"),
    AGY("agy", List.of("-p"), "agy -p", "");

    private final String binary;
    private final List<String> leadingArguments;
    private final String label;
    private final String defaultModel;

    AgentCli(String binary, List<String> leadingArguments, String label, String defaultModel) {
        this.binary = binary;
        this.leadingArguments = leadingArguments;
        this.label = label;
        this.defaultModel = defaultModel;
    }

    public String binary() {
        return binary;
    }

    public List<String> leadingArguments() {
        return leadingArguments;
    }

    public String label() {
        return label;
    }

    public String defaultModel() {
        return defaultModel;
    }

    public String wireName() {
        return name().toLowerCase();
    }

    public static AgentCli of(String value) {
        return Arrays.stream(values())
                .filter(cli -> cli.wireName().equalsIgnoreCase(value))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unsupported agent cli: " + value));
    }
}
