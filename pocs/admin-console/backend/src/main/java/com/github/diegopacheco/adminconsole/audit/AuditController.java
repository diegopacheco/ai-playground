package com.github.diegopacheco.adminconsole.audit;

import io.swagger.v3.oas.annotations.Operation;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/audit")
public class AuditController {
    private final AuditRepository repository;

    public AuditController(AuditRepository repository) {
        this.repository = repository;
    }

    @GetMapping
    @Operation(summary = "Search the audit trail")
    public Map<String, Object> search(@RequestParam(required = false) String user,
                                      @RequestParam(required = false) Long connection,
                                      @RequestParam(required = false) Boolean allowed,
                                      @RequestParam(required = false) String from,
                                      @RequestParam(required = false) String to,
                                      @RequestParam(defaultValue = "1") int page,
                                      @RequestParam(defaultValue = "100") int size) {
        int limit = Math.min(Math.max(size, 1), 500);
        int offset = Math.max(page - 1, 0) * limit;
        List<AuditEntry> entries = repository.search(user, connection, allowed, instant(from), instant(to), limit, offset);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("page", page);
        body.put("size", limit);
        body.put("entries", entries.stream().map(AuditController::view).toList());
        return body;
    }

    @GetMapping(value = "/export.csv", produces = MediaType.TEXT_PLAIN_VALUE)
    @Operation(summary = "Export the audit trail as CSV")
    public String export(@RequestParam(required = false) String user,
                         @RequestParam(required = false) Long connection,
                         @RequestParam(required = false) Boolean allowed) {
        StringBuilder csv = new StringBuilder("at,username,connection_id,kind,allowed,denial_reason,elapsed_ms,row_count,statement\n");
        for (AuditEntry entry : repository.search(user, connection, allowed, null, null, 500, 0)) {
            csv.append(entry.at()).append(',')
                    .append(entry.username()).append(',')
                    .append(entry.connectionId()).append(',')
                    .append(entry.kind()).append(',')
                    .append(entry.allowed()).append(',')
                    .append(quote(entry.denialReason())).append(',')
                    .append(entry.elapsedMs()).append(',')
                    .append(entry.rowCount()).append(',')
                    .append(quote(entry.statement())).append('\n');
        }
        return csv.toString();
    }

    private static String quote(String value) {
        if (value == null) {
            return "";
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private static Instant instant(String value) {
        return value == null || value.isBlank() ? null : Instant.parse(value);
    }

    private static Map<String, Object> view(AuditEntry entry) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", entry.id());
        body.put("queryId", entry.queryId());
        body.put("page", entry.page());
        body.put("at", entry.at());
        body.put("username", entry.username());
        body.put("connectionId", entry.connectionId());
        body.put("kind", entry.kind());
        body.put("statement", entry.statement());
        body.put("allowed", entry.allowed());
        body.put("denialReason", entry.denialReason());
        body.put("elapsedMs", entry.elapsedMs());
        body.put("rowCount", entry.rowCount());
        body.put("error", entry.error());
        return body;
    }
}
