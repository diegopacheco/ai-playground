package io.factoryfile;

import java.util.List;
import java.util.Optional;

public record Factory(
        String name,
        Optional<String> description,
        Optional<String> entry,
        List<Agent> agents,
        List<Stage> stages,
        List<Loop> loops) {
    public Factory {
        agents = List.copyOf(agents);
        stages = List.copyOf(stages);
        loops = List.copyOf(loops);
    }

    public Optional<Agent> agent(String name) {
        return agents.stream().filter(a -> a.name().equals(name)).findFirst();
    }

    public Optional<Stage> stage(String name) {
        return stages.stream().filter(s -> s.name().equals(name)).findFirst();
    }
}
