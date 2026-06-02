package com.linter.site;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.nio.file.Path;
import java.nio.file.Paths;

@Component
public class LintPaths {

    private final Path repoDir;
    private final Path lintDir;

    public LintPaths(
            @Value("${lint.repoDir:/data/repo}") String repoDir,
            @Value("${lint.lintDir:/data/lint}") String lintDir) {
        this.repoDir = Paths.get(repoDir).toAbsolutePath().normalize();
        this.lintDir = Paths.get(lintDir).toAbsolutePath().normalize();
    }

    public Path repoDir() {
        return repoDir;
    }

    public Path lintDir() {
        return lintDir;
    }

    public Path resolveInRepo(String relative) {
        Path resolved = repoDir.resolve(relative).toAbsolutePath().normalize();
        if (!resolved.startsWith(repoDir)) {
            throw new IllegalArgumentException("path escapes repo root");
        }
        return resolved;
    }
}
