package io.factoryfile;

import java.util.Optional;

public record Loop(
        String name,
        Optional<String> description,
        String from,
        String to,
        Optional<String> until,
        Optional<Integer> max) {
}
