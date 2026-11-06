package io.agentsfile;

import java.util.Optional;

public record Command(String name, String source, Optional<String> description) {
}
