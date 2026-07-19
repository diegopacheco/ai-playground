package com.github.diegopacheco.devadminconsole.saved;

import com.github.diegopacheco.devadminconsole.auth.CurrentUser;
import com.github.diegopacheco.devadminconsole.project.ConnectionRepository;
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
@RequestMapping("/api/projects/{projectId}/saved")
public class SavedQueryController {
    public record SaveRequest(@NotBlank String name, @NotBlank String statement, Long connectionId,
                              String description) {}

    private final SavedQueryRepository saved;
    private final ConnectionRepository connections;
    private final CurrentUser current;

    public SavedQueryController(SavedQueryRepository saved, ConnectionRepository connections, CurrentUser current) {
        this.saved = saved;
        this.connections = connections;
        this.current = current;
    }

    @GetMapping
    @Operation(summary = "List saved queries shared in a project")
    public List<Map<String, Object>> list(@PathVariable long projectId) {
        return saved.findByProject(projectId).stream().map(SavedQueryController::view).toList();
    }

    @PostMapping
    @Operation(summary = "Save a query for the whole project")
    public Map<String, Object> create(@PathVariable long projectId, @RequestBody SaveRequest request) {
        String kind = kindOf(request.connectionId());
        return view(saved.create(new SavedQuery(null, projectId, request.connectionId(), request.name(),
                request.statement(), kind, request.description(), current.username(), null, null)));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Edit a saved query")
    public Map<String, Object> update(@PathVariable long projectId, @PathVariable long id,
                                      @RequestBody SaveRequest request) {
        saved.findById(id).orElseThrow(() -> new IllegalArgumentException("saved query not found"));
        saved.update(id, request.name(), request.statement(), request.description(), request.connectionId());
        return Map.of("updated", true);
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a saved query")
    public Map<String, Object> delete(@PathVariable long projectId, @PathVariable long id) {
        saved.delete(id);
        return Map.of("deleted", true);
    }

    private String kindOf(Long connectionId) {
        if (connectionId == null) {
            return "any";
        }
        return connections.findById(connectionId)
                .orElseThrow(() -> new IllegalArgumentException("connection not found"))
                .kind().wireName();
    }

    private static Map<String, Object> view(SavedQuery query) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", query.id());
        body.put("projectId", query.projectId());
        body.put("connectionId", query.connectionId());
        body.put("name", query.name());
        body.put("statement", query.statement());
        body.put("kind", query.kind());
        body.put("description", query.description());
        body.put("createdBy", query.createdBy());
        body.put("updatedAt", query.updatedAt());
        return body;
    }
}
