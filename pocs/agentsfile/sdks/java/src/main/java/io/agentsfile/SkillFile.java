package io.agentsfile;

import java.util.Optional;

public record SkillFile(String path, Optional<String> source, Optional<String> content) {
}
