package com.github.diegopacheco.devadminconsole.crypto;

import java.security.SecureRandom;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
public class PostgresKeyStore implements MasterKeyProvider {
    public static final String MASTER = "master";
    public static final String JWT = "jwt";
    private static final int KEY_BYTES = 32;

    private final JdbcTemplate jdbc;
    private final SecureRandom random = new SecureRandom();
    private final Map<String, byte[]> cache = new ConcurrentHashMap<>();

    public PostgresKeyStore(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public byte[] key(String purpose) {
        return cache.computeIfAbsent(purpose, this::load).clone();
    }

    private byte[] load(String purpose) {
        byte[] candidate = new byte[KEY_BYTES];
        random.nextBytes(candidate);
        jdbc.update("INSERT INTO keys (purpose, key_material) VALUES (?, ?) ON CONFLICT (purpose) DO NOTHING", purpose, candidate);
        return jdbc.queryForObject("SELECT key_material FROM keys WHERE purpose = ?", byte[].class, purpose);
    }
}
