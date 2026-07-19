package com.github.diegopacheco.adminconsole.registry;

import com.datastax.oss.driver.api.core.CqlSession;
import com.datastax.oss.driver.api.core.CqlSessionBuilder;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import io.etcd.jetcd.Client;
import io.lettuce.core.RedisClient;
import io.lettuce.core.RedisURI;
import io.lettuce.core.api.StatefulRedisConnection;
import jakarta.annotation.PreDestroy;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import javax.sql.DataSource;
import org.apache.kafka.clients.admin.Admin;
import org.apache.kafka.clients.admin.AdminClientConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class ConnectionRegistry {
    private static final Logger log = LoggerFactory.getLogger(ConnectionRegistry.class);

    private record Entry(AutoCloseable resource, AtomicLong lastUsed) {}

    private final Map<Long, Entry> resources = new ConcurrentHashMap<>();
    private final int maximumPoolSize;
    private final long connectionTimeoutMs;
    private final long idleEvictionSeconds;

    public ConnectionRegistry(@Value("${app.pool.maximum-size}") int maximumPoolSize,
                              @Value("${app.pool.connection-timeout-ms}") long connectionTimeoutMs,
                              @Value("${app.pool.idle-eviction-seconds}") long idleEvictionSeconds) {
        this.maximumPoolSize = maximumPoolSize;
        this.connectionTimeoutMs = connectionTimeoutMs;
        this.idleEvictionSeconds = idleEvictionSeconds;
    }

    public DataSource jdbc(ConnectionConfig config) {
        return (DataSource) resource(config, () -> buildJdbc(config));
    }

    public CqlSession cassandra(ConnectionConfig config) {
        return (CqlSession) resource(config, () -> buildCassandra(config));
    }

    public StatefulRedisConnection<String, String> redis(ConnectionConfig config) {
        return ((RedisHandle) resource(config, () -> buildRedis(config))).connection();
    }

    public Client etcd(ConnectionConfig config) {
        return (Client) resource(config, () -> buildEtcd(config));
    }

    public Admin kafka(ConnectionConfig config) {
        return (Admin) resource(config, () -> buildKafka(config));
    }

    public void evict(Long connectionId) {
        Entry entry = resources.remove(connectionId);
        if (entry != null) {
            close(entry.resource());
        }
    }

    @Scheduled(fixedDelay = 60_000)
    public void evictIdle() {
        long threshold = System.currentTimeMillis() - idleEvictionSeconds * 1000;
        resources.forEach((id, entry) -> {
            if (entry.lastUsed().get() < threshold) {
                evict(id);
            }
        });
    }

    @PreDestroy
    public void closeAll() {
        resources.keySet().forEach(this::evict);
    }

    private AutoCloseable resource(ConnectionConfig config, java.util.function.Supplier<AutoCloseable> factory) {
        Entry entry = resources.computeIfAbsent(config.id(),
                id -> new Entry(factory.get(), new AtomicLong(System.currentTimeMillis())));
        entry.lastUsed().set(System.currentTimeMillis());
        return entry.resource();
    }

    private AutoCloseable buildJdbc(ConnectionConfig config) {
        HikariConfig hikari = new HikariConfig();
        hikari.setJdbcUrl(jdbcUrl(config));
        hikari.setUsername(config.username());
        hikari.setPassword(config.password());
        hikari.setMaximumPoolSize(maximumPoolSize);
        hikari.setMinimumIdle(0);
        hikari.setConnectionTimeout(connectionTimeoutMs);
        hikari.setInitializationFailTimeout(-1);
        hikari.setReadOnly(true);
        hikari.setPoolName("admin-console-" + config.id());
        if (config.kind() == com.github.diegopacheco.adminconsole.project.ConnectionKind.POSTGRES) {
            Properties properties = new Properties();
            properties.setProperty("options", "-c default_transaction_read_only=on");
            hikari.setDataSourceProperties(properties);
        }
        return new HikariDataSource(hikari);
    }

    private String jdbcUrl(ConnectionConfig config) {
        return switch (config.kind()) {
            case MYSQL -> "jdbc:mysql://" + config.host() + ":" + config.port() + "/"
                    + (config.database() == null ? "" : config.database());
            case POSTGRES -> "jdbc:postgresql://" + config.host() + ":" + config.port() + "/"
                    + (config.database() == null ? "postgres" : config.database());
            default -> throw new IllegalArgumentException("not a jdbc connection: " + config.kind().wireName());
        };
    }

    private AutoCloseable buildCassandra(ConnectionConfig config) {
        CqlSessionBuilder builder = CqlSession.builder()
                .addContactPoint(new InetSocketAddress(config.host(), config.port()))
                .withLocalDatacenter(config.datacenter() == null ? "datacenter1" : config.datacenter());
        if (config.username() != null && !config.username().isBlank()) {
            builder.withAuthCredentials(config.username(), config.password() == null ? "" : config.password());
        }
        if (config.keyspace() != null && !config.keyspace().isBlank()) {
            builder.withKeyspace(config.keyspace());
        }
        return builder.build();
    }

    private AutoCloseable buildRedis(ConnectionConfig config) {
        RedisURI.Builder uri = RedisURI.builder().withHost(config.host()).withPort(config.port())
                .withTimeout(Duration.ofMillis(connectionTimeoutMs));
        if (config.password() != null && !config.password().isEmpty()) {
            uri.withPassword(config.password().toCharArray());
        }
        if (config.database() != null && !config.database().isBlank()) {
            uri.withDatabase(Integer.parseInt(config.database()));
        }
        RedisClient client = RedisClient.create(uri.build());
        StatefulRedisConnection<String, String> connection = client.connect();
        return new RedisHandle(client, connection);
    }

    private AutoCloseable buildEtcd(ConnectionConfig config) {
        return Client.builder().endpoints("http://" + config.host() + ":" + config.port()).build();
    }

    private AutoCloseable buildKafka(ConnectionConfig config) {
        Properties properties = new Properties();
        properties.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, config.host() + ":" + config.port());
        properties.put(AdminClientConfig.REQUEST_TIMEOUT_MS_CONFIG, (int) connectionTimeoutMs);
        properties.put(AdminClientConfig.CLIENT_ID_CONFIG, "admin-console-" + config.id());
        return Admin.create(properties);
    }

    private void close(AutoCloseable resource) {
        try {
            resource.close();
        } catch (Exception error) {
            log.warn("connection resource did not close cleanly: {}", error.getMessage());
        }
    }
}
