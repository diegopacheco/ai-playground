package com.game.terminator.agent;

import java.util.List;
import java.util.Map;

public class AgentRegistry {

    public record AgentInfo(String name, List<String> models) {}

    public static final List<AgentInfo> AVAILABLE_AGENTS = List.of(
        new AgentInfo("claude", List.of("opus", "sonnet", "haiku")),
        new AgentInfo("gemini", List.of("gemini-3.1-pro", "gemini-3-flash", "gemini-2.5-pro")),
        new AgentInfo("copilot", List.of("claude-sonnet-4.6", "claude-sonnet-4.5", "gemini-3-pro")),
        new AgentInfo("codex", List.of("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"))
    );

    public static List<Map<String, Object>> toJson() {
        return AVAILABLE_AGENTS.stream().map(a -> Map.<String, Object>of(
            "name", a.name(),
            "models", a.models()
        )).toList();
    }
}
