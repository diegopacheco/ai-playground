package com.github.diegopacheco.devadminconsole.discovery;

import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@Component
public class ContainerScanner {
    public record Runtime(String binary, boolean available) {}

    private static final List<String> RUNTIMES = List.of("podman", "docker");
    private static final int MAX_OUTPUT_BYTES = 4 * 1024 * 1024;

    private final EngineDetector detector;
    private final ObjectMapper mapper = new ObjectMapper();
    private final long timeoutSeconds;

    private final int ownPort;

    public ContainerScanner(EngineDetector detector,
                            @Value("${app.discovery.timeout-seconds:20}") long timeoutSeconds,
                            @Value("${spring.datasource.url}") String ownDatasourceUrl) {
        this.detector = detector;
        this.timeoutSeconds = timeoutSeconds;
        this.ownPort = portOf(ownDatasourceUrl);
    }

    static int portOf(String jdbcUrl) {
        if (jdbcUrl == null) {
            return 0;
        }
        int lastColon = jdbcUrl.lastIndexOf(':');
        if (lastColon < 0) {
            return 0;
        }
        StringBuilder digits = new StringBuilder();
        for (int index = lastColon + 1; index < jdbcUrl.length(); index++) {
            char character = jdbcUrl.charAt(index);
            if (!Character.isDigit(character)) {
                break;
            }
            digits.append(character);
        }
        return digits.isEmpty() ? 0 : Integer.parseInt(digits.toString());
    }

    public Runtime runtime() {
        for (String binary : RUNTIMES) {
            try {
                if (run(List.of(binary, "info", "--format", "{{.Host.Arch}}")) != null) {
                    return new Runtime(binary, true);
                }
            } catch (RuntimeException ignored) {
                continue;
            }
        }
        return new Runtime(null, false);
    }

    public List<DiscoveredContainer> scan() {
        Runtime runtime = runtime();
        if (!runtime.available()) {
            throw new IllegalStateException("neither podman nor docker is available to the console process");
        }
        JsonNode containers = mapper.readTree(run(List.of(runtime.binary(), "ps", "--format", "json")));
        List<DiscoveredContainer> found = new ArrayList<>();
        for (JsonNode container : containers) {
            String image = text(container, "Image");
            Optional<ConnectionKind> kind = detector.detect(image);
            if (kind.isEmpty()) {
                continue;
            }
            String name = name(container);
            String id = text(container, "Id");
            int containerPort = detector.defaultPort(kind.get());
            int hostPort = hostPort(container, containerPort);
            if (hostPort == 0) {
                found.add(DiscoveredContainer.unreachable(shortId(id), name, image, kind.get(), containerPort,
                        "no published host port — the console cannot reach it from outside the container network"));
                continue;
            }
            if (hostPort == ownPort) {
                found.add(DiscoveredContainer.unreachable(shortId(id), name, image, kind.get(), containerPort,
                        "this is the console's own metadata database — importing it would expose the encryption "
                                + "key, password hashes and every stored connection secret"));
                continue;
            }
            Map<String, String> environment = environment(runtime.binary(), id);
            found.add(new DiscoveredContainer(shortId(id), name, image, kind.get(), hostPort, containerPort,
                    detector.database(kind.get(), environment),
                    detector.keyspace(kind.get(), environment),
                    detector.username(kind.get(), environment),
                    detector.password(kind.get(), environment),
                    true,
                    detector.isSuperuser(kind.get(), environment)
                            ? "the detected credentials are the container's superuser"
                            : null));
        }
        found.sort((left, right) -> left.name().compareTo(right.name()));
        return found;
    }

    private Map<String, String> environment(String binary, String id) {
        Map<String, String> values = new LinkedHashMap<>();
        try {
            JsonNode inspected = mapper.readTree(run(List.of(binary, "inspect", id, "--format", "json")));
            JsonNode node = inspected.isArray() ? inspected.get(0) : inspected;
            JsonNode env = node.path("Config").path("Env");
            for (JsonNode entry : env) {
                String line = entry.asString();
                int separator = line.indexOf('=');
                if (separator > 0) {
                    values.put(line.substring(0, separator), line.substring(separator + 1));
                }
            }
        } catch (RuntimeException ignored) {
            return values;
        }
        return values;
    }

    private int hostPort(JsonNode container, int containerPort) {
        JsonNode ports = container.path("Ports");
        int fallback = 0;
        for (JsonNode port : ports) {
            int host = port.path("host_port").asInt(0);
            int inside = port.path("container_port").asInt(0);
            if (host == 0) {
                continue;
            }
            if (inside == containerPort) {
                return host;
            }
            if (fallback == 0) {
                fallback = host;
            }
        }
        return fallback;
    }

    private String name(JsonNode container) {
        JsonNode names = container.path("Names");
        if (names.isArray() && !names.isEmpty()) {
            return names.get(0).asString();
        }
        return names.isMissingNode() ? text(container, "Name") : names.asString();
    }

    private String text(JsonNode container, String field) {
        JsonNode value = container.path(field);
        return value.isMissingNode() || value.isNull() ? "" : value.asString();
    }

    private String shortId(String id) {
        return id == null || id.length() <= 12 ? id : id.substring(0, 12);
    }

    private String run(List<String> command) {
        Process process = null;
        try {
            ProcessBuilder builder = new ProcessBuilder(command);
            builder.redirectErrorStream(false);
            process = builder.start();
            byte[] bytes = process.getInputStream().readNBytes(MAX_OUTPUT_BYTES);
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException(command.getFirst() + " did not answer in " + timeoutSeconds + "s");
            }
            if (process.exitValue() != 0) {
                throw new IllegalStateException(command.getFirst() + " exited with " + process.exitValue());
            }
            return new String(bytes, StandardCharsets.UTF_8);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(command.getFirst() + " was interrupted", error);
        } catch (IOException error) {
            throw new IllegalStateException(command.getFirst() + " is not available: " + error.getMessage(), error);
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
            }
        }
    }
}
