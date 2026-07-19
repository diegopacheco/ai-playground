package com.github.diegopacheco.devadminconsole.federation;

import com.github.diegopacheco.devadminconsole.engine.EngineRegistry;
import com.github.diegopacheco.devadminconsole.engine.PageRequest;
import com.github.diegopacheco.devadminconsole.engine.QueryResult;
import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import com.github.diegopacheco.devadminconsole.project.ConnectionConfig;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;

@Service
public class FederatedExampleBuilder {
    private record Candidate(ConnectionConfig connection, String source, List<String> columns,
                             Map<String, Set<String>> values) {}

    private static final int SAMPLE_ROWS = 40;
    private static final int MAX_CANDIDATES = 10;

    private final EngineRegistry engines;
    private final FederatedQueryParser parser;
    private final FederatedExecutor executor;

    public FederatedExampleBuilder(EngineRegistry engines, FederatedQueryParser parser, FederatedExecutor executor) {
        this.engines = engines;
        this.parser = parser;
        this.executor = executor;
    }

    public String build(List<ConnectionConfig> connections) {
        List<Candidate> candidates = sample(connections);
        if (candidates.size() < 2) {
            return null;
        }

        List<String> aliases = List.of("a", "b", "c", "d");
        Candidate base = null;
        List<String> clauses = new ArrayList<>();
        List<String> projection = new ArrayList<>();
        Set<String> usedConnections = new LinkedHashSet<>();

        for (Candidate candidate : candidates) {
            if (base == null) {
                base = candidate;
                usedConnections.add(candidate.connection().name());
                projection.add("a." + firstReadable(candidate));
                continue;
            }
            if (clauses.size() >= 3 || usedConnections.contains(candidate.connection().name())) {
                continue;
            }
            String[] pair = sharedKey(base, candidate);
            if (pair == null) {
                continue;
            }
            String alias = aliases.get(clauses.size() + 1);
            clauses.add("JOIN " + candidate.connection().name() + "." + candidate.source() + " " + alias
                    + " ON a." + pair[0] + " = " + alias + "." + pair[1]);
            projection.add(alias + "." + firstReadable(candidate));
            usedConnections.add(candidate.connection().name());
        }

        if (clauses.isEmpty()) {
            return null;
        }

        String statement = "SELECT " + String.join(", ", projection) + "\n"
                + "FROM " + base.connection().name() + "." + base.source() + " a\n"
                + String.join("\n", clauses) + "\n"
                + "LIMIT 25";

        return returnsRows(statement, connections) ? statement : null;
    }

    private boolean returnsRows(String statement, List<ConnectionConfig> connections) {
        try {
            return !executor.execute(parser.parse(statement), connections).rows().isEmpty();
        } catch (RuntimeException error) {
            return false;
        }
    }

    private String[] sharedKey(Candidate left, Candidate right) {
        for (String leftColumn : left.columns()) {
            Set<String> leftValues = left.values().get(leftColumn);
            if (leftValues == null || leftValues.isEmpty()) {
                continue;
            }
            for (String rightColumn : right.columns()) {
                Set<String> rightValues = right.values().get(rightColumn);
                if (rightValues == null || rightValues.isEmpty()) {
                    continue;
                }
                if (leftValues.stream().anyMatch(rightValues::contains)) {
                    return new String[] {leftColumn, rightColumn};
                }
            }
        }
        return null;
    }

    private String firstReadable(Candidate candidate) {
        return candidate.columns().stream()
                .filter(column -> !column.startsWith("_"))
                .findFirst()
                .orElse(candidate.columns().isEmpty() ? "*" : candidate.columns().getFirst());
    }

    private List<Candidate> sample(List<ConnectionConfig> connections) {
        List<Candidate> candidates = new ArrayList<>();
        for (ConnectionConfig connection : connections) {
            if (candidates.size() >= MAX_CANDIDATES) {
                break;
            }
            try {
                for (SchemaNode node : engines.of(connection.kind()).schema(connection)) {
                    if (candidates.size() >= MAX_CANDIDATES) {
                        break;
                    }
                    FederatedQuery.Side side = new FederatedQuery.Side("s", connection.name(), node.name(), null);
                    QueryResult rows = engines.of(connection.kind()).query(connection,
                            executor.nativeStatement(connection.kind(), side), PageRequest.first(SAMPLE_ROWS));
                    if (rows.rows().isEmpty()) {
                        continue;
                    }
                    Map<String, Set<String>> values = new LinkedHashMap<>();
                    for (String column : rows.columns()) {
                        Set<String> seen = new LinkedHashSet<>();
                        for (Map<String, Object> row : rows.rows()) {
                            Object value = row.get(column);
                            if (value != null) {
                                seen.add(String.valueOf(value));
                            }
                        }
                        values.put(column, seen);
                    }
                    candidates.add(new Candidate(connection, node.name(), rows.columns(), values));
                    break;
                }
            } catch (RuntimeException ignored) {
                continue;
            }
        }
        return candidates;
    }
}
