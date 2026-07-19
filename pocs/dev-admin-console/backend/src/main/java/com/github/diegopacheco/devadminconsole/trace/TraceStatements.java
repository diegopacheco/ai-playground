package com.github.diegopacheco.devadminconsole.trace;

import com.github.diegopacheco.devadminconsole.engine.SchemaNode;
import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

@Component
public class TraceStatements {
    public record Probe(String source, String statement) {}

    private static final List<String> KEYISH = List.of("id", "key", "uuid", "sku", "code", "number", "email", "name");

    public List<Probe> probes(ConnectionKind kind, List<SchemaNode> schema, String term, TraceBudget budget) {
        return switch (kind) {
            case POSTGRES, MYSQL -> sqlProbes(schema, term, budget);
            case CASSANDRA -> cassandraProbes(schema, term, budget);
            case REDIS -> redisProbes(term, budget);
            case ETCD -> etcdProbes(term);
            case KAFKA -> kafkaProbes(schema, budget);
            case ELASTICSEARCH -> elasticProbes(schema, term, budget);
        };
    }

    private List<Probe> sqlProbes(List<SchemaNode> schema, String term, TraceBudget budget) {
        List<Probe> probes = new ArrayList<>();
        for (SchemaNode table : schema) {
            if (probes.size() >= budget.sourcesPerConnection()) {
                break;
            }
            List<String> candidates = candidateColumns(table);
            if (candidates.isEmpty()) {
                continue;
            }
            String where = candidates.stream()
                    .map(column -> "CAST(" + column + " AS CHAR(255)) = '" + escape(term) + "'")
                    .reduce((left, right) -> left + " OR " + right)
                    .orElseThrow();
            probes.add(new Probe(table.name(),
                    "SELECT * FROM " + table.name() + " WHERE " + where + " LIMIT " + budget.rowsPerSource()));
        }
        return probes;
    }

    private List<Probe> cassandraProbes(List<SchemaNode> schema, String term, TraceBudget budget) {
        List<Probe> probes = new ArrayList<>();
        for (SchemaNode table : schema) {
            if (probes.size() >= budget.sourcesPerConnection()) {
                break;
            }
            Optional<SchemaNode> partitionKey = table.children().stream()
                    .filter(column -> column.detail() != null && column.detail().contains("partition_key"))
                    .filter(column -> typeAccepts(column.detail(), term))
                    .findFirst();
            partitionKey.ifPresent(key -> probes.add(new Probe(table.name(),
                    "SELECT * FROM " + table.name() + " WHERE " + key.name() + " = " + literal(term)
                            + " LIMIT " + budget.rowsPerSource())));
        }
        return probes;
    }

    private List<Probe> redisProbes(String term, TraceBudget budget) {
        return List.of(
                new Probe("key " + term, "TYPE " + term),
                new Probe("keys matching *" + term + "*",
                        "SCAN 0 MATCH *" + term + "* COUNT " + budget.scanWindow()));
    }

    private List<Probe> etcdProbes(String term) {
        return List.of(
                new Probe("key " + term, "get " + term),
                new Probe("prefix " + term, "get " + term + " --prefix"));
    }

    private List<Probe> kafkaProbes(List<SchemaNode> schema, TraceBudget budget) {
        List<Probe> probes = new ArrayList<>();
        for (SchemaNode topic : schema) {
            if (probes.size() >= budget.sourcesPerConnection()) {
                break;
            }
            probes.add(new Probe(topic.name(), "consume " + topic.name() + " --limit " + budget.scanWindow()));
        }
        return probes;
    }

    private List<Probe> elasticProbes(List<SchemaNode> schema, String term, TraceBudget budget) {
        List<Probe> probes = new ArrayList<>();
        for (SchemaNode index : schema) {
            if (probes.size() >= budget.sourcesPerConnection()) {
                break;
            }
            probes.add(new Probe(index.name(),
                    "GET /" + index.name() + "/_search {\"query\":{\"query_string\":{\"query\":\""
                            + escapeJson(term) + "\"}}}"));
        }
        return probes;
    }

    List<String> candidateColumns(SchemaNode table) {
        List<String> candidates = new ArrayList<>();
        for (SchemaNode column : table.children()) {
            String name = column.name().toLowerCase();
            String type = column.detail() == null ? "" : column.detail().toLowerCase();
            boolean keyish = KEYISH.stream().anyMatch(name::contains);
            boolean textish = type.contains("char") || type.contains("text") || type.contains("uuid");
            if (keyish || textish) {
                candidates.add(column.name());
            }
        }
        return candidates;
    }

    boolean typeAccepts(String detail, String term) {
        String type = detail.toLowerCase();
        boolean numeric = term.chars().allMatch(Character::isDigit);
        boolean uuid = term.matches("[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");
        if (type.contains("uuid")) {
            return uuid;
        }
        if (type.contains("int") || type.contains("decimal") || type.contains("float") || type.contains("double")) {
            return numeric;
        }
        if (type.contains("text") || type.contains("varchar") || type.contains("ascii")) {
            return !uuid;
        }
        return false;
    }

    String escape(String term) {
        return term.replace("'", "''");
    }

    String escapeJson(String term) {
        return term.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    String literal(String term) {
        return term.chars().allMatch(Character::isDigit) ? term : "'" + escape(term) + "'";
    }
}
