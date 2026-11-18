package com.github.diegopacheco.restate.agents;

import com.github.diegopacheco.restate.AgentException;
import com.github.diegopacheco.restate.config.AgentConfig;
import org.springframework.stereotype.Component;

@Component
public class AgentCaller {

    private final CommandRunner runner;

    public AgentCaller(CommandRunner runner) {
        this.runner = runner;
    }

    public String ask(AgentConfig config, String prompt) {
        if (config.model().isBlank()) {
            throw new AgentException("model is required");
        }
        if (prompt.isBlank()) {
            throw new AgentException("prompt is required");
        }
        return runner.run(config.provider().command(config.model(), prompt)).trim();
    }
}
