package com.github.diegopacheco.devadminconsole.engine.elastic;

import org.springframework.stereotype.Component;

@Component
public class ElasticCommandParser {
    public ElasticRequest parse(String command) {
        if (command == null || command.isBlank()) {
            throw new IllegalArgumentException("command is required");
        }
        String trimmed = command.trim();
        int firstSpace = trimmed.indexOf(' ');
        if (firstSpace < 0) {
            throw new IllegalArgumentException("expected a method and a path, for example: GET /index/_search");
        }
        String method = trimmed.substring(0, firstSpace).toUpperCase();
        String remainder = trimmed.substring(firstSpace + 1).trim();
        int bodyStart = remainder.indexOf('{');
        String path;
        String body = null;
        if (bodyStart >= 0) {
            path = remainder.substring(0, bodyStart).trim();
            body = remainder.substring(bodyStart).trim();
        } else {
            path = remainder;
        }
        if (path.isEmpty()) {
            throw new IllegalArgumentException("a path is required");
        }
        if (!path.startsWith("/")) {
            path = "/" + path;
        }
        return new ElasticRequest(method, path, body);
    }
}
