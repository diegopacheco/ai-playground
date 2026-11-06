package io.agentsfile;

import java.util.List;
import java.util.Optional;

public record Skill(
        String name,
        Optional<String> description,
        Optional<String> source,
        Optional<String> content,
        List<SkillFile> files) {
    public Skill {
        files = List.copyOf(files);
    }
}
