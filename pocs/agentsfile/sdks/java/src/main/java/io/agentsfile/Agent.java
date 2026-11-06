package io.agentsfile;

import java.util.List;
import java.util.Optional;

public record Agent(
        String name,
        Optional<String> description,
        Optional<String> model,
        List<String> tools,
        Optional<String> source,
        Optional<String> content) {
    public Agent {
        tools = List.copyOf(tools);
    }
}
