package io.agentsfile;

import java.util.List;
import java.util.Optional;

public record Command(
        String name,
        Optional<String> description,
        Optional<String> argumentHint,
        Optional<String> model,
        List<String> tools,
        Optional<String> source,
        Optional<String> content) {
    public Command {
        tools = List.copyOf(tools);
    }
}
