package com.github.diegopacheco.adminconsole.trace;

import com.github.diegopacheco.adminconsole.project.ConnectionRepository;
import io.swagger.v3.oas.annotations.Operation;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/projects/{projectId}/trace")
public class TraceController {
    private final EntityTracer tracer;
    private final ConnectionRepository connections;

    public TraceController(EntityTracer tracer, ConnectionRepository connections) {
        this.tracer = tracer;
        this.connections = connections;
    }

    @GetMapping
    @Operation(summary = "Find a value across every connection in a project")
    public Map<String, Object> trace(@PathVariable long projectId, @RequestParam String term) {
        EntityTracer.Trace trace = tracer.trace(connections.findByProject(projectId), term, TraceBudget.defaults());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("term", trace.term());
        body.put("elapsedMs", trace.elapsedMs());
        body.put("truncated", trace.truncated());
        body.put("hits", trace.hits().stream().map(TraceController::view).toList());
        body.put("failures", trace.failures().stream().map(failure -> Map.of(
                "connectionName", failure.connectionName(),
                "kind", failure.kind(),
                "reason", failure.reason())).toList());
        return body;
    }

    private static Map<String, Object> view(TraceHit hit) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("connectionName", hit.connectionName());
        body.put("kind", hit.kind());
        body.put("source", hit.source());
        body.put("label", hit.label());
        body.put("at", hit.at());
        body.put("columns", hit.columns());
        body.put("row", hit.row());
        return body;
    }
}
