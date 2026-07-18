package com.github.diegopacheco.adminconsole.federation;

import com.github.diegopacheco.adminconsole.audit.AuditService;
import com.github.diegopacheco.adminconsole.auth.CurrentUser;
import com.github.diegopacheco.adminconsole.project.ConnectionRepository;
import jakarta.servlet.http.HttpServletRequest;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/projects/{projectId}/federated")
public class FederationController {
    public record FederatedRequest(@NotBlank String statement) {}

    private final FederatedQueryParser parser;
    private final FederatedExecutor executor;
    private final ConnectionRepository connections;
    private final AuditService audit;
    private final CurrentUser current;

    public FederationController(FederatedQueryParser parser, FederatedExecutor executor,
                                ConnectionRepository connections, AuditService audit, CurrentUser current) {
        this.parser = parser;
        this.executor = executor;
        this.connections = connections;
        this.audit = audit;
        this.current = current;
    }

    @org.springframework.web.bind.annotation.GetMapping("/history")
    @Operation(summary = "Recent cross-engine joins run by the current user in this project")
    public java.util.List<String> history(@PathVariable long projectId,
                                          @org.springframework.web.bind.annotation.RequestParam(defaultValue = "20")
                                          int limit) {
        return audit.recentFederated(current.username(), projectId, Math.min(Math.max(limit, 1), 100));
    }

    @PostMapping
    @Operation(summary = "Join two sources that live on different engines")
    public Map<String, Object> query(@PathVariable long projectId, @RequestBody FederatedRequest request,
                                     HttpServletRequest servletRequest) {
        FederatedQuery query = parser.parse(request.statement());
        java.util.UUID queryId = java.util.UUID.randomUUID();
        FederatedExecutor.Result result;
        try {
            result = executor.execute(query, connections.findByProject(projectId));
        } catch (RuntimeException error) {
            audit.federated(queryId, current.username(), projectId, request.statement(), 0, null,
                    error.getMessage(), servletRequest.getRemoteAddr());
            throw error;
        }
        audit.federated(queryId, current.username(), projectId, request.statement(), result.elapsedMs(),
                result.rows().size(), null, servletRequest.getRemoteAddr());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("columns", result.columns());
        body.put("rows", result.rows());
        body.put("elapsedMs", result.elapsedMs());
        body.put("sides", result.sides().stream().map(side -> {
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("alias", side.alias());
            view.put("connectionName", side.connectionName());
            view.put("kind", side.kind());
            view.put("source", side.source());
            view.put("rows", side.rows());
            view.put("truncated", side.truncated());
            return view;
        }).toList());
        return body;
    }
}
