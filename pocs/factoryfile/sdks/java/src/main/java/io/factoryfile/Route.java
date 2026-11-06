package io.factoryfile;

import java.util.Optional;

public record Route(
        String outcome,
        String to,
        Optional<String> label,
        Optional<String> reason,
        boolean cycle,
        Optional<Exhausted> exhausted) {
}
