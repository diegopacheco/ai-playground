package com.github.diegopacheco.devadminconsole.trace;

import com.github.diegopacheco.devadminconsole.engine.Engine;
import com.github.diegopacheco.devadminconsole.engine.EngineRegistry;
import com.github.diegopacheco.devadminconsole.engine.PageRequest;
import com.github.diegopacheco.devadminconsole.engine.QueryResult;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.springframework.stereotype.Service;

@Service
public class EntityTracer {
    public record Failure(String connectionName, String kind, String reason) {}

    public record Trace(String term, List<TraceHit> hits, List<Failure> failures, long elapsedMs, boolean truncated) {}

    private final EngineRegistry engines;
    private final TraceStatements statements;
    private final TraceTimestamps timestamps;

    public EntityTracer(EngineRegistry engines, TraceStatements statements, TraceTimestamps timestamps) {
        this.engines = engines;
        this.statements = statements;
        this.timestamps = timestamps;
    }

    public Trace trace(List<ConnectionConfig> connections, String term, TraceBudget budget) {
        if (term == null || term.isBlank()) {
            throw new IllegalArgumentException("what should I look for?");
        }
        long started = System.currentTimeMillis();
        List<TraceHit> hits = new CopyOnWriteArrayList<>();
        List<Failure> failures = new CopyOnWriteArrayList<>();

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<Void>> tasks = new ArrayList<>();
            for (ConnectionConfig connection : connections) {
                tasks.add(executor.submit(task(connection, term.trim(), budget, hits, failures)));
            }
            for (int index = 0; index < tasks.size(); index++) {
                try {
                    tasks.get(index).get(budget.perConnectionTimeoutSeconds() + 2, TimeUnit.SECONDS);
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception error) {
                    ConnectionConfig connection = connections.get(index);
                    failures.add(new Failure(connection.name(), connection.kind().wireName(),
                            "timed out after " + budget.perConnectionTimeoutSeconds() + "s"));
                }
            }
        }

        List<TraceHit> ordered = new ArrayList<>(hits);
        ordered.sort(Comparator.comparing(TraceHit::at, Comparator.nullsLast(Comparator.reverseOrder())));
        boolean truncated = ordered.size() > budget.totalHits();
        if (truncated) {
            ordered = new ArrayList<>(ordered.subList(0, budget.totalHits()));
        }
        return new Trace(term.trim(), ordered, new ArrayList<>(failures),
                System.currentTimeMillis() - started, truncated);
    }

    private Callable<Void> task(ConnectionConfig connection, String term, TraceBudget budget,
                                List<TraceHit> hits, List<Failure> failures) {
        return () -> {
            Engine engine = engines.of(connection.kind());
            List<TraceStatements.Probe> probes;
            try {
                probes = statements.probes(connection.kind(), engine.schema(connection), term, budget);
            } catch (RuntimeException error) {
                failures.add(new Failure(connection.name(), connection.kind().wireName(),
                        "schema unavailable: " + message(error)));
                return null;
            }
            if (probes.isEmpty()) {
                failures.add(new Failure(connection.name(), connection.kind().wireName(),
                        connection.kind() == ConnectionKind.CASSANDRA
                                ? "no partition key matches this term — a non-key scan needs ALLOW FILTERING, "
                                        + "which scans the whole cluster, so it is not attempted"
                                : "nothing searchable on this connection"));
                return null;
            }
            for (TraceStatements.Probe probe : probes) {
                try {
                    QueryResult result = engine.query(connection, probe.statement(),
                            PageRequest.first(scansAndFilters(connection.kind())
                                    ? budget.scanWindow()
                                    : budget.rowsPerSource()));
                    collect(connection, probe, result, term, hits, budget);
                } catch (RuntimeException error) {
                    failures.add(new Failure(connection.name(), connection.kind().wireName(),
                            probe.source() + ": " + message(error)));
                }
            }
            return null;
        };
    }

    private void collect(ConnectionConfig connection, TraceStatements.Probe probe, QueryResult result, String term,
                         List<TraceHit> hits, TraceBudget budget) {
        int kept = 0;
        for (Map<String, Object> row : result.rows()) {
            if (kept >= budget.rowsPerSource()) {
                return;
            }
            if (!matches(connection.kind(), row, term)) {
                continue;
            }
            Instant at = timestamps.of(row);
            hits.add(new TraceHit(connection.name(), connection.kind().wireName(), probe.source(),
                    label(row), at, result.columns(), row));
            kept++;
        }
    }

    static boolean scansAndFilters(ConnectionKind kind) {
        return kind == ConnectionKind.KAFKA || kind == ConnectionKind.REDIS || kind == ConnectionKind.ETCD;
    }

    private boolean matches(ConnectionKind kind, Map<String, Object> row, String term) {
        if (!scansAndFilters(kind)) {
            return true;
        }
        String needle = term.toLowerCase();
        return row.values().stream()
                .filter(value -> value != null)
                .anyMatch(value -> String.valueOf(value).toLowerCase().contains(needle));
    }

    private String label(Map<String, Object> row) {
        return row.entrySet().stream()
                .filter(entry -> entry.getValue() != null)
                .limit(2)
                .map(entry -> entry.getKey() + "=" + String.valueOf(entry.getValue()))
                .reduce((left, right) -> left + " · " + right)
                .orElse("");
    }

    private String message(RuntimeException error) {
        return error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
    }
}
