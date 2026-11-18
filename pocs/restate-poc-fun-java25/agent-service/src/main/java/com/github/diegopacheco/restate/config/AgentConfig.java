package com.github.diegopacheco.restate.config;

import com.github.diegopacheco.restate.AgentException;
import com.github.diegopacheco.restate.agents.Provider;

import java.util.Optional;
import java.util.function.Function;

public record AgentConfig(Provider provider, String model) {

    public static AgentConfig from(Function<String, String> env) {
        String raw = Optional.ofNullable(env.apply("AGENT_PROVIDER")).orElse("");
        Provider provider = Provider.parse(raw)
                .orElseThrow(() -> new AgentException("AGENT_PROVIDER must be claude, codex, agy or ollama, got \"" + raw + "\""));
        String model = Optional.ofNullable(env.apply("AGENT_MODEL")).orElse("");
        if (model.isBlank()) {
            throw new AgentException("AGENT_MODEL is required");
        }
        return new AgentConfig(provider, model);
    }
}
