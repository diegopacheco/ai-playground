package com.github.diegopacheco.adminconsole.registry;

import io.lettuce.core.RedisClient;
import io.lettuce.core.api.StatefulRedisConnection;

public record RedisHandle(RedisClient client, StatefulRedisConnection<String, String> connection) implements AutoCloseable {
    @Override
    public void close() {
        connection.close();
        client.shutdown();
    }
}
