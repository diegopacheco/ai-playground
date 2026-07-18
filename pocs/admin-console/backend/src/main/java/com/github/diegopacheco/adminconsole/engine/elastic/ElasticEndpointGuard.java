package com.github.diegopacheco.adminconsole.engine.elastic;

import com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation;
import java.util.List;
import java.util.Set;
import org.springframework.stereotype.Component;

@Component
public class ElasticEndpointGuard {
    private static final Set<String> ALLOWED_METHODS = Set.of("GET", "HEAD");
    private static final List<String> DENIED_ENDPOINTS = List.of(
            "_bulk", "_update_by_query", "_delete_by_query", "_reindex", "_close", "_open", "_forcemerge",
            "_refresh", "_flush", "_shrink", "_split", "_clone", "_rollover", "_freeze", "_unfreeze",
            "_snapshot", "_restore", "_cache/clear", "_scripts", "_ingest", "_ilm", "_slm", "_security",
            "_delete", "_create", "_update", "_doc");

    public void assertReadOnly(ElasticRequest request) {
        if (!ALLOWED_METHODS.contains(request.method())) {
            throw new ReadOnlyViolation(request.method() + " is not allowed, only GET and HEAD are read operations");
        }
        String path = request.path().toLowerCase();
        for (String denied : DENIED_ENDPOINTS) {
            if (path.contains("/" + denied) || path.endsWith("/" + denied)) {
                throw new ReadOnlyViolation(denied + " can modify the cluster and is not allowed, even as a GET");
            }
        }
    }
}
