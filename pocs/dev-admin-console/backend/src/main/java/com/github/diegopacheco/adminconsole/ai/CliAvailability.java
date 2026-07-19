package com.github.diegopacheco.adminconsole.ai;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.stereotype.Component;

@Component
public class CliAvailability {
    public record Availability(boolean available, String reason, String path) {}

    public Map<AgentCli, Availability> all() {
        Map<AgentCli, Availability> result = new LinkedHashMap<>();
        for (AgentCli cli : AgentCli.values()) {
            result.put(cli, of(cli));
        }
        return result;
    }

    public Availability of(AgentCli cli) {
        String pathVariable = System.getenv("PATH");
        if (pathVariable == null || pathVariable.isBlank()) {
            return new Availability(false, "PATH is not set for the console process", null);
        }
        for (String entry : pathVariable.split(java.io.File.pathSeparator)) {
            if (entry.isBlank()) {
                continue;
            }
            Path candidate = Path.of(entry, cli.binary());
            if (Files.isRegularFile(candidate) && Files.isExecutable(candidate)) {
                return new Availability(true, null, candidate.toString());
            }
        }
        return new Availability(false, cli.binary() + " was not found on PATH", null);
    }
}
