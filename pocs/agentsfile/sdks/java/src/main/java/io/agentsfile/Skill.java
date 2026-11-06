package io.agentsfile;

import java.util.Optional;

public record Skill(String name, String source, Optional<String> description) {
}
