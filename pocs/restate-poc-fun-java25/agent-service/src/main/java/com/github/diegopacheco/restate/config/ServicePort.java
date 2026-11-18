package com.github.diegopacheco.restate.config;

import com.github.diegopacheco.restate.AgentException;

import java.util.function.Function;

public final class ServicePort {

    private ServicePort() {
    }

    public static int from(Function<String, String> env) {
        try {
            int port = Integer.parseInt(env.apply("AGENT_SERVICE_PORT"));
            if (port > 0 && port <= 65535) {
                return port;
            }
        } catch (NumberFormatException ignored) {
        }
        throw new AgentException("AGENT_SERVICE_PORT is required");
    }
}
