package io.factoryfile;

import java.util.Optional;

public record Exhausted(
        String to,
        Optional<String> label,
        Optional<String> reason,
        Optional<Exhausted> exhausted) {
}
