package com.linter.site;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

@RestController
@RequestMapping("/api")
@CrossOrigin
public class ReportController {

    private static final java.util.Set<String> SKIP = java.util.Set.of(
            "node_modules", "target", "dist", "build", ".git", ".lint", ".idea", ".vite");

    private final LintPaths paths;

    public ReportController(LintPaths paths) {
        this.paths = paths;
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok");
    }

    @GetMapping(value = "/report", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> report() {
        Path file = paths.lintDir().resolve("report.json");
        if (!Files.exists(file)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "report.json not found; run /lint first");
        }
        return ResponseEntity.ok(readString(file));
    }

    @GetMapping(value = "/history", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> history() {
        Path dir = paths.lintDir().resolve("history");
        if (!Files.isDirectory(dir)) {
            return ResponseEntity.ok("[]");
        }
        try (Stream<Path> files = Files.list(dir)) {
            List<String> entries = files
                    .filter(p -> p.getFileName().toString().endsWith(".json"))
                    .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                    .map(this::readString)
                    .toList();
            return ResponseEntity.ok("[" + String.join(",", entries) + "]");
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
        }
    }

    @GetMapping(value = "/tree", produces = MediaType.APPLICATION_JSON_VALUE)
    public List<Map<String, Object>> tree() {
        List<Map<String, Object>> files = new ArrayList<>();
        collect(paths.repoDir(), files);
        files.sort(Comparator.comparing(f -> (String) f.get("path")));
        return files;
    }

    @GetMapping(value = "/source", produces = MediaType.TEXT_PLAIN_VALUE)
    public ResponseEntity<String> source(@RequestParam String path) {
        Path file = paths.resolveInRepo(path);
        if (!Files.isRegularFile(file)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "file not found");
        }
        return ResponseEntity.ok(readString(file));
    }

    private void collect(Path dir, List<Map<String, Object>> out) {
        try (Stream<Path> stream = Files.list(dir)) {
            for (Path p : stream.toList()) {
                String name = p.getFileName().toString();
                if (Files.isDirectory(p)) {
                    if (!SKIP.contains(name)) {
                        collect(p, out);
                    }
                } else if (isSource(name)) {
                    out.add(Map.of(
                            "path", paths.repoDir().relativize(p).toString(),
                            "loc", countLines(p)));
                }
            }
        } catch (IOException ignored) {
        }
    }

    private boolean isSource(String name) {
        return name.endsWith(".java") || name.endsWith(".js") || name.endsWith(".jsx")
                || name.endsWith(".ts") || name.endsWith(".tsx") || name.endsWith(".mjs")
                || name.endsWith(".css") || name.endsWith(".json") || name.endsWith(".xml")
                || name.endsWith(".properties") || name.endsWith(".md") || name.endsWith(".sh");
    }

    private int countLines(Path p) {
        try {
            return (int) Files.lines(p).count();
        } catch (IOException e) {
            return 0;
        }
    }

    private String readString(Path p) {
        try {
            return Files.readString(p);
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
        }
    }
}
