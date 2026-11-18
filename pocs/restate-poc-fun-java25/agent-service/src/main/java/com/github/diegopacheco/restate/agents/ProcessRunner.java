package com.github.diegopacheco.restate.agents;

import com.github.diegopacheco.restate.AgentException;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Stream;

@Component
public class ProcessRunner implements CommandRunner {

    @Override
    public String run(List<String> command) {
        if (command.isEmpty()) {
            throw new AgentException("empty command");
        }
        Path home = Path.of(Optional.ofNullable(System.getenv("HOME")).orElse(""));
        ProcessBuilder builder = new ProcessBuilder(command).redirectInput(ProcessBuilder.Redirect.from(new File("/dev/null")));
        builder.environment().put("PATH", commandPath(home, Optional.ofNullable(System.getenv("PATH")).orElse("")));
        try {
            Process process = builder.start();
            CompletableFuture<String> stderr = CompletableFuture.supplyAsync(() -> read(process.getErrorStream()));
            String stdout = read(process.getInputStream());
            int code = process.waitFor();
            if (code == 0) {
                return stdout;
            }
            String detail = stderr.join().isBlank() ? stdout : stderr.join();
            throw new AgentException("Command failed with exit status " + code + ": " + detail.trim());
        } catch (IOException error) {
            throw new AgentException("Unable to start " + command.getFirst() + ": " + error.getMessage());
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new AgentException("Interrupted while running " + command.getFirst());
        }
    }

    private static String read(java.io.InputStream stream) {
        try (stream) {
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException error) {
            return "";
        }
    }

    public static String commandPath(Path home, String current) {
        List<Path> candidates = new ArrayList<>(List.of(
                home.resolve(".local/bin"),
                home.resolve(".bun/bin"),
                Path.of("/opt/homebrew/bin"),
                Path.of("/usr/local/bin")));
        candidates.addAll(nodeBinaries(home));
        Arrays.stream(current.split(File.pathSeparator)).filter(part -> !part.isEmpty()).map(Path::of).forEach(candidates::add);
        return String.join(File.pathSeparator, new LinkedHashSet<>(candidates.stream().map(Path::toString).toList()));
    }

    private static List<Path> nodeBinaries(Path home) {
        Path versions = home.resolve(".nvm/versions/node");
        if (!Files.isDirectory(versions)) {
            return List.of();
        }
        try (Stream<Path> entries = Files.list(versions)) {
            return entries.sorted(Comparator.reverseOrder()).map(entry -> entry.resolve("bin")).toList();
        } catch (IOException error) {
            return List.of();
        }
    }
}
