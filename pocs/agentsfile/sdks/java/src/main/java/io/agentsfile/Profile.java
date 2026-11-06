package io.agentsfile;

import java.util.List;
import java.util.Optional;

public record Profile(
        String name,
        Optional<String> description,
        List<Skill> skills,
        List<Agent> agents,
        List<Command> commands,
        List<Mcp> mcps) {
    public Profile {
        skills = List.copyOf(skills);
        agents = List.copyOf(agents);
        commands = List.copyOf(commands);
        mcps = List.copyOf(mcps);
    }
}
