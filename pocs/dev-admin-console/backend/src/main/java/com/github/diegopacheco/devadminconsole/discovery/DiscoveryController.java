package com.github.diegopacheco.devadminconsole.discovery;

import com.github.diegopacheco.devadminconsole.auth.CurrentUser;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import com.github.diegopacheco.devadminconsole.project.ConnectionRepository;
import com.github.diegopacheco.devadminconsole.project.Project;
import com.github.diegopacheco.devadminconsole.project.ProjectRepository;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/discovery")
public class DiscoveryController {
    public record ImportRequest(@NotBlank String projectName, List<String> containerIds) {}

    private final ContainerScanner scanner;
    private final ProjectRepository projects;
    private final ConnectionRepository connections;
    private final CurrentUser current;

    public DiscoveryController(ContainerScanner scanner, ProjectRepository projects,
                               ConnectionRepository connections, CurrentUser current) {
        this.scanner = scanner;
        this.projects = projects;
        this.connections = connections;
        this.current = current;
    }

    @GetMapping
    @Operation(summary = "List running containers that look like a supported engine")
    public Map<String, Object> scan() {
        ContainerScanner.Runtime runtime = scanner.runtime();
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("runtime", runtime.binary());
        if (!runtime.available()) {
            body.put("available", false);
            body.put("reason", "neither podman nor docker is available to the console process");
            body.put("containers", List.of());
            return body;
        }
        body.put("available", true);
        body.put("containers", scanner.scan().stream().map(DiscoveryController::view).toList());
        return body;
    }

    @PostMapping("/import")
    @Operation(summary = "Import selected containers as a new project")
    public Map<String, Object> importContainers(@RequestBody ImportRequest request) {
        if (request.containerIds() == null || request.containerIds().isEmpty()) {
            throw new IllegalArgumentException("select at least one container to import");
        }
        List<DiscoveredContainer> found = scanner.scan();
        List<DiscoveredContainer> chosen = found.stream()
                .filter(container -> request.containerIds().contains(container.id()))
                .filter(DiscoveredContainer::importable)
                .toList();
        if (chosen.isEmpty()) {
            throw new IllegalArgumentException("none of the selected containers are still running and importable");
        }
        Project project = projects.create(request.projectName(), current.username());
        List<String> imported = new ArrayList<>();
        for (DiscoveredContainer container : chosen) {
            connections.create(new ConnectionConfig(null, project.id(), container.name(), container.kind(),
                    "localhost", container.hostPort(), container.database(), container.keyspace(),
                    container.kind() == ConnectionKind.CASSANDRA ? "datacenter1" : null,
                    container.username(), container.password(), null, current.username()));
            imported.add(container.name());
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("projectId", project.id());
        body.put("projectName", project.name());
        body.put("imported", imported);
        return body;
    }

    private static Map<String, Object> view(DiscoveredContainer container) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", container.id());
        body.put("name", container.name());
        body.put("image", container.image());
        body.put("kind", container.kind().wireName());
        body.put("hostPort", container.hostPort());
        body.put("containerPort", container.containerPort());
        body.put("database", container.database());
        body.put("keyspace", container.keyspace());
        body.put("username", container.username());
        body.put("hasPassword", container.password() != null && !container.password().isEmpty());
        body.put("importable", container.importable());
        body.put("reason", container.reason());
        return body;
    }
}
