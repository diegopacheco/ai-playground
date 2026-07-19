package com.github.diegopacheco.adminconsole.project;

import java.util.Arrays;

public enum ConnectionKind {
    MYSQL(3306),
    POSTGRES(5432),
    CASSANDRA(9042),
    REDIS(6379),
    ETCD(2379),
    KAFKA(9092),
    ELASTICSEARCH(9200);

    private final int defaultPort;

    ConnectionKind(int defaultPort) {
        this.defaultPort = defaultPort;
    }

    public int defaultPort() {
        return defaultPort;
    }

    public String wireName() {
        return name().toLowerCase();
    }

    public static ConnectionKind of(String value) {
        return Arrays.stream(values())
                .filter(kind -> kind.wireName().equalsIgnoreCase(value))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unsupported connection kind: " + value));
    }
}
