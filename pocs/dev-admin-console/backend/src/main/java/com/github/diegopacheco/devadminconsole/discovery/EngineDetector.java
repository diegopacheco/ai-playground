package com.github.diegopacheco.devadminconsole.discovery;

import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.stereotype.Component;

@Component
public class EngineDetector {
    private record Signature(ConnectionKind kind, List<String> imageTokens, int defaultPort) {}

    private static final List<Signature> SIGNATURES = List.of(
            new Signature(ConnectionKind.POSTGRES, List.of("postgres", "postgresql", "timescale", "pgvector"), 5432),
            new Signature(ConnectionKind.MYSQL, List.of("mysql", "mariadb", "percona"), 3306),
            new Signature(ConnectionKind.CASSANDRA, List.of("cassandra", "scylla"), 9042),
            new Signature(ConnectionKind.REDIS, List.of("redis", "valkey", "keydb"), 6379),
            new Signature(ConnectionKind.ETCD, List.of("etcd"), 2379),
            new Signature(ConnectionKind.KAFKA, List.of("kafka", "redpanda"), 9092),
            new Signature(ConnectionKind.ELASTICSEARCH, List.of("elasticsearch", "opensearch"), 9200));

    public Optional<ConnectionKind> detect(String image) {
        if (image == null || image.isBlank()) {
            return Optional.empty();
        }
        String name = image.toLowerCase();
        int tagSeparator = name.lastIndexOf(':');
        if (tagSeparator > 0) {
            name = name.substring(0, tagSeparator);
        }
        String repository = name.substring(name.lastIndexOf('/') + 1);
        for (Signature signature : SIGNATURES) {
            for (String token : signature.imageTokens()) {
                if (repository.equals(token) || repository.startsWith(token + "-") || repository.endsWith("-" + token)
                        || repository.contains(token)) {
                    return Optional.of(signature.kind());
                }
            }
        }
        return Optional.empty();
    }

    public int defaultPort(ConnectionKind kind) {
        return SIGNATURES.stream()
                .filter(signature -> signature.kind() == kind)
                .findFirst()
                .map(Signature::defaultPort)
                .orElse(kind.defaultPort());
    }

    public String database(ConnectionKind kind, Map<String, String> environment) {
        return switch (kind) {
            case POSTGRES -> environment.get("POSTGRES_DB");
            case MYSQL -> environment.getOrDefault("MYSQL_DATABASE", environment.get("MARIADB_DATABASE"));
            default -> null;
        };
    }

    public String keyspace(ConnectionKind kind, Map<String, String> environment) {
        return kind == ConnectionKind.POSTGRES ? "public" : null;
    }

    public String username(ConnectionKind kind, Map<String, String> environment) {
        return switch (kind) {
            case POSTGRES -> environment.getOrDefault("POSTGRES_USER", "postgres");
            case MYSQL -> environment.containsKey("MYSQL_USER") ? environment.get("MYSQL_USER") : "root";
            case ELASTICSEARCH -> environment.containsKey("ELASTIC_PASSWORD") ? "elastic" : null;
            default -> null;
        };
    }

    public String password(ConnectionKind kind, Map<String, String> environment) {
        return switch (kind) {
            case POSTGRES -> environment.get("POSTGRES_PASSWORD");
            case MYSQL -> environment.containsKey("MYSQL_USER")
                    ? environment.get("MYSQL_PASSWORD")
                    : environment.getOrDefault("MYSQL_ROOT_PASSWORD", environment.get("MARIADB_ROOT_PASSWORD"));
            case ELASTICSEARCH -> environment.get("ELASTIC_PASSWORD");
            case REDIS -> environment.get("REDIS_PASSWORD");
            default -> null;
        };
    }

    public boolean isSuperuser(ConnectionKind kind, Map<String, String> environment) {
        return switch (kind) {
            case POSTGRES -> !environment.containsKey("POSTGRES_USER")
                    || "postgres".equals(environment.get("POSTGRES_USER"));
            case MYSQL -> !environment.containsKey("MYSQL_USER");
            case ELASTICSEARCH -> environment.containsKey("ELASTIC_PASSWORD");
            default -> false;
        };
    }
}
