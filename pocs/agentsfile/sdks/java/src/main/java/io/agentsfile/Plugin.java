package io.agentsfile;

import java.util.List;
import java.util.Optional;

public record Plugin(
        String name,
        Optional<String> description,
        Optional<String> version,
        Optional<String> author,
        Optional<String> source,
        List<Skill> skills,
        List<Agent> agents,
        List<Command> commands,
        List<Mcp> mcps) {
    public Plugin {
        skills = List.copyOf(skills);
        agents = List.copyOf(agents);
        commands = List.copyOf(commands);
        mcps = List.copyOf(mcps);
    }
}
