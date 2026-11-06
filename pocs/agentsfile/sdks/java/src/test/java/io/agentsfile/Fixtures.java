package io.agentsfile;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

final class Fixtures {
    private Fixtures() {
    }

    static Path root() {
        Path dir = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        while (dir != null) {
            Path candidate = dir.resolve("spec").resolve("fixtures");
            if (Files.isDirectory(candidate)) {
                return candidate;
            }
            dir = dir.getParent();
        }
        throw new IllegalStateException("spec/fixtures not found above " + System.getProperty("user.dir"));
    }

    static Path valid(String name) {
        return root().resolve("valid").resolve(name);
    }

    static List<Path> invalid() {
        try (Stream<Path> files = Files.list(root().resolve("invalid"))) {
            return files.filter(p -> p.toString().endsWith(".agentsfile")).sorted().toList();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
