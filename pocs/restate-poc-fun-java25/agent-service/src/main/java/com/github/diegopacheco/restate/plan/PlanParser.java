package com.github.diegopacheco.restate.plan;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.diegopacheco.restate.AgentException;

import java.io.IOException;
import java.util.List;
import java.util.stream.StreamSupport;

public final class PlanParser {

    public static final int MAX_TASKS = 4;
    private static final ObjectMapper JSON = new ObjectMapper();

    private PlanParser() {
    }

    public static Plan parse(String output) {
        int start = output.indexOf('[');
        int end = output.lastIndexOf(']');
        if (start < 0 || end <= start) {
            String preview = output.substring(0, Math.min(200, output.length()));
            throw new AgentException("agent returned no JSON task list: " + preview);
        }
        JsonNode values = read(output.substring(start, end + 1));
        List<String> tasks = StreamSupport.stream(values.spliterator(), false)
                .filter(JsonNode::isTextual)
                .map(node -> node.asText().trim())
                .filter(task -> !task.isEmpty())
                .limit(MAX_TASKS)
                .toList();
        if (tasks.isEmpty()) {
            throw new AgentException("agent plan has no tasks");
        }
        return new Plan(tasks);
    }

    private static JsonNode read(String json) {
        try {
            JsonNode node = JSON.readTree(json);
            if (!node.isArray()) {
                throw new AgentException("agent plan is not a list");
            }
            return node;
        } catch (IOException error) {
            throw new AgentException("agent plan is not a list: " + error.getMessage());
        }
    }
}
