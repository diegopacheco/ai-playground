package io.factoryfile;

import java.util.List;
import java.util.Optional;

public record Stage(
        String name,
        Optional<String> description,
        List<String> agents,
        List<String> next,
        List<String> artifacts,
        Optional<String> content) {
    public Stage {
        agents = List.copyOf(agents);
        next = List.copyOf(next);
        artifacts = List.copyOf(artifacts);
    }
}
