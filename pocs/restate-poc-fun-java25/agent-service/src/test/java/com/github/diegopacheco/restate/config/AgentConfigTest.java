package com.github.diegopacheco.restate.config;

import com.github.diegopacheco.restate.AgentException;
import com.github.diegopacheco.restate.agents.Provider;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AgentConfigTest {

    @Test
    void readsProviderAndModelFromAgentConf() {
        AgentConfig config = AgentConfig.from(Map.of("AGENT_PROVIDER", "ollama", "AGENT_MODEL", "llama3.2")::get);
        assertEquals(new AgentConfig(Provider.OLLAMA, "llama3.2"), config);
    }

    @Test
    void rejectsAnUnknownProviderAtStartupInsteadOfFailingMidRun() {
        AgentException error = assertThrows(AgentException.class, () -> AgentConfig.from(Map.of("AGENT_PROVIDER", "gpt", "AGENT_MODEL", "m")::get));
        assertEquals("AGENT_PROVIDER must be claude, codex, agy or ollama, got \"gpt\"", error.getMessage());
    }

    @Test
    void requiresAModel() {
        assertThrows(AgentException.class, () -> AgentConfig.from(Map.of("AGENT_PROVIDER", "claude")::get));
    }

    @Test
    void requiresAValidServicePort() {
        assertEquals(9280, ServicePort.from(Map.of("AGENT_SERVICE_PORT", "9280")::get));
        assertThrows(AgentException.class, () -> ServicePort.from(Map.of("AGENT_SERVICE_PORT", "abc")::get));
        assertThrows(AgentException.class, () -> ServicePort.from(Map.<String, String>of()::get));
    }
}
