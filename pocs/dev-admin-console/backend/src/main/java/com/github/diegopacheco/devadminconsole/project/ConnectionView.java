package com.github.diegopacheco.devadminconsole.project;

import java.util.LinkedHashMap;
import java.util.Map;

public final class ConnectionView {
    private ConnectionView() {
    }

    public static Map<String, Object> of(ConnectionConfig config) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", config.id());
        body.put("projectId", config.projectId());
        body.put("name", config.name());
        body.put("kind", config.kind().wireName());
        body.put("host", config.host());
        body.put("port", config.port());
        body.put("database", config.database());
        body.put("keyspace", config.keyspace());
        body.put("datacenter", config.datacenter());
        body.put("username", config.username());
        body.put("hasPassword", config.password() != null && !config.password().isEmpty());
        body.put("createdBy", config.createdBy());
        return body;
    }
}
