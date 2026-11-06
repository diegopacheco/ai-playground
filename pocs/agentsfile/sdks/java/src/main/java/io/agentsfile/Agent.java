package io.agentsfile;

import java.util.List;
import java.util.Optional;

public record Agent(String name, String source, Optional<String> description, Optional<String> model, List<String> tools) {
    public Agent {
        tools = List.copyOf(tools);
    }
}
