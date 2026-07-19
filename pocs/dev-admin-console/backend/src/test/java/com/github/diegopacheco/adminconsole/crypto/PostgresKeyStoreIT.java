package com.github.diegopacheco.adminconsole.crypto;

import static org.assertj.core.api.Assertions.assertThat;

import javax.sql.DataSource;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

@Tag("integration-test")
@SpringBootTest
class PostgresKeyStoreIT {
    @Autowired
    private DataSource dataSource;

    private JdbcTemplate jdbc;

    @BeforeEach
    void setUp() {
        jdbc = new JdbcTemplate(dataSource);
        jdbc.update("DELETE FROM keys WHERE purpose = 'test-purpose'");
    }

    @Test
    void generatesAKeyOnceAndReturnsTheSameKeyAfterRestartSoStoredSecretsStayDecryptable() {
        byte[] first = new PostgresKeyStore(jdbc).key("test-purpose");
        byte[] afterRestart = new PostgresKeyStore(jdbc).key("test-purpose");
        assertThat(first).hasSize(32).isEqualTo(afterRestart);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM keys WHERE purpose = 'test-purpose'", Integer.class)).isEqualTo(1);
    }

    @Test
    void concurrentBootsAgreeOnOneKeySoParallelStartupCannotOrphanEncryptedSecrets() throws Exception {
        int racers = 8;
        var results = new java.util.concurrent.CopyOnWriteArrayList<String>();
        var latch = new java.util.concurrent.CountDownLatch(racers);
        try (var executor = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
            for (int index = 0; index < racers; index++) {
                executor.submit(() -> {
                    latch.countDown();
                    latch.await();
                    results.add(java.util.HexFormat.of().formatHex(new PostgresKeyStore(jdbc).key("test-purpose")));
                    return null;
                });
            }
        }
        assertThat(results).hasSize(racers);
        assertThat(java.util.Set.copyOf(results)).hasSize(1);
    }

    @Test
    void keepsDistinctKeysPerPurposeSoRotatingOneDoesNotInvalidateTheOther() {
        PostgresKeyStore store = new PostgresKeyStore(jdbc);
        assertThat(store.key(PostgresKeyStore.MASTER)).isNotEqualTo(store.key(PostgresKeyStore.JWT));
    }

    @org.junit.jupiter.api.AfterEach
    void removeTheTestKey() {
        jdbc.update("DELETE FROM keys WHERE purpose = 'test-purpose'");
    }
}
