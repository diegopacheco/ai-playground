package com.github.diegopacheco.devadminconsole.project;

import com.github.diegopacheco.devadminconsole.auth.CurrentUser;
import com.github.diegopacheco.devadminconsole.registry.ConnectionRegistry;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {
    public record ProjectRequest(@NotBlank String name) {}

    public record ConnectionRequest(@NotBlank String name, @NotBlank String kind, @NotBlank String host, Integer port,
                                    String database, String keyspace, String datacenter, String username,
                                    String password, String options) {}

    private final ProjectRepository projects;
    private final ConnectionRepository connections;
    private final ConnectionRegistry registry;
    private final CurrentUser current;

    public ProjectController(ProjectRepository projects, ConnectionRepository connections,
                             ConnectionRegistry registry, CurrentUser current) {
        this.projects = projects;
        this.connections = connections;
        this.registry = registry;
        this.current = current;
    }

    @GetMapping
    @Operation(summary = "List projects with their connections")
    public List<Map<String, Object>> list() {
        return projects.findAll().stream().map(project -> {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("id", project.id());
            body.put("name", project.name());
            body.put("createdBy", project.createdBy());
            body.put("connections", connections.findByProject(project.id()).stream().map(ConnectionView::of).toList());
            return body;
        }).toList();
    }

    @PostMapping
    @Operation(summary = "Create a project")
    public Map<String, Object> create(@RequestBody ProjectRequest request) {
        Project project = projects.create(request.name(), current.username());
        return Map.of("id", project.id(), "name", project.name());
    }

    @PutMapping("/{id}")
    @Operation(summary = "Rename a project")
    public Map<String, Object> rename(@PathVariable long id, @RequestBody ProjectRequest request) {
        projects.rename(id, request.name());
        return Map.of("updated", true);
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a project and its connections")
    public Map<String, Object> delete(@PathVariable long id) {
        connections.findByProject(id).forEach(connection -> registry.evict(connection.id()));
        projects.delete(id);
        return Map.of("deleted", true);
    }

    @PostMapping("/{id}/connections")
    @Operation(summary = "Add a connection to a project")
    public Map<String, Object> addConnection(@PathVariable long id, @RequestBody ConnectionRequest request) {
        projects.findById(id).orElseThrow(() -> new IllegalArgumentException("project not found"));
        return ConnectionView.of(connections.create(toConfig(null, id, request)));
    }

    @PutMapping("/{id}/connections/{connectionId}")
    @Operation(summary = "Edit a connection")
    public Map<String, Object> editConnection(@PathVariable long id, @PathVariable long connectionId,
                                              @RequestBody ConnectionRequest request) {
        connections.findById(connectionId).orElseThrow(() -> new IllegalArgumentException("connection not found"));
        boolean replacePassword = request.password() != null;
        connections.update(connectionId, toConfig(connectionId, id, request), replacePassword);
        registry.evict(connectionId);
        return Map.of("updated", true);
    }

    @DeleteMapping("/{id}/connections/{connectionId}")
    @Operation(summary = "Remove a connection")
    public Map<String, Object> deleteConnection(@PathVariable long id, @PathVariable long connectionId) {
        registry.evict(connectionId);
        connections.delete(connectionId);
        return Map.of("deleted", true);
    }

    private ConnectionConfig toConfig(Long id, long projectId, ConnectionRequest request) {
        ConnectionKind kind = ConnectionKind.of(request.kind());
        int port = request.port() == null ? kind.defaultPort() : request.port();
        return new ConnectionConfig(id, projectId, request.name(), kind, request.host(), port, request.database(),
                request.keyspace(), request.datacenter(), request.username(), request.password(), request.options(),
                current.username());
    }
}
