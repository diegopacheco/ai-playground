package com.github.diegopacheco.adminconsole.project;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import javax.sql.DataSource;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

@Tag("integration-test")
@SpringBootTest
class ConnectionRepositoryIT {
    @Autowired
    private ConnectionRepository connections;
    @Autowired
    private ProjectRepository projects;
    @Autowired
    private DataSource dataSource;

    private JdbcTemplate jdbc;
    private long projectId;

    @BeforeEach
    void setUp() {
        jdbc = new JdbcTemplate(dataSource);
        jdbc.update("DELETE FROM connections WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM projects WHERE name LIKE 'it-%'");
        projectId = projects.create("it-project-" + System.nanoTime(), "tester").id();
    }

    private ConnectionConfig config(String name, String password) {
        return new ConnectionConfig(null, projectId, name, ConnectionKind.POSTGRES, "localhost", 5432,
                "shop", null, null, "postgres", password, null, "tester");
    }

    @Test
    void roundTripsAPasswordSoTheConsoleCanActuallyConnectAfterARestart() {
        ConnectionConfig created = connections.create(config("it-roundtrip", "sup3rs3cret"));
        assertThat(connections.findById(created.id()).orElseThrow().password()).isEqualTo("sup3rs3cret");
    }

    @Test
    void neverStoresThePasswordInReadableFormSoAGlanceAtTheTableRevealsNothing() {
        ConnectionConfig created = connections.create(config("it-ciphertext", "sup3rs3cret"));
        byte[] stored = jdbc.queryForObject("SELECT secret_ciphertext FROM connections WHERE id = ?",
                byte[].class, created.id());
        assertThat(new String(stored, StandardCharsets.UTF_8)).doesNotContain("sup3rs3cret");
        assertThat(stored).isNotNull();
    }

    @Test
    void encryptsTheSamePasswordDifferentlyPerRowSoReusedPasswordsAreNotCorrelatable() {
        ConnectionConfig first = connections.create(config("it-iv-one", "identical"));
        ConnectionConfig second = connections.create(config("it-iv-two", "identical"));
        byte[] a = jdbc.queryForObject("SELECT secret_ciphertext FROM connections WHERE id = ?", byte[].class, first.id());
        byte[] b = jdbc.queryForObject("SELECT secret_ciphertext FROM connections WHERE id = ?", byte[].class, second.id());
        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void keepsTheExistingPasswordWhenAnEditOmitsItSoRenamingAConnectionDoesNotBreakIt() {
        ConnectionConfig created = connections.create(config("it-keep", "original"));
        ConnectionConfig edit = new ConnectionConfig(created.id(), projectId, "it-keep-renamed", ConnectionKind.POSTGRES,
                "localhost", 5432, "shop", null, null, "postgres", null, null, "tester");
        connections.update(created.id(), edit, false);
        ConnectionConfig reloaded = connections.findById(created.id()).orElseThrow();
        assertThat(reloaded.name()).isEqualTo("it-keep-renamed");
        assertThat(reloaded.password()).isEqualTo("original");
    }

    @Test
    void replacesThePasswordWhenAnEditSuppliesOne() {
        ConnectionConfig created = connections.create(config("it-replace", "original"));
        connections.update(created.id(), config("it-replace", "rotated").withPassword("rotated"), true);
        assertThat(connections.findById(created.id()).orElseThrow().password()).isEqualTo("rotated");
    }

    @Test
    void storesConnectionsWithoutAPasswordSoUnauthenticatedTargetsAreSupported() {
        ConnectionConfig created = connections.create(config("it-nopassword", null));
        assertThat(connections.findById(created.id()).orElseThrow().password()).isNull();
    }

    @Test
    void deletesConnectionsWithTheirProjectSoNoOrphanedCredentialsSurvive() {
        connections.create(config("it-cascade", "secret"));
        projects.delete(projectId);
        assertThat(connections.findByProject(projectId)).isEmpty();
    }

    @org.junit.jupiter.api.AfterEach
    void cleanUpSoTestDataNeverShowsUpInSomeonesConsole() {
        jdbc.update("DELETE FROM saved_queries WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM connections WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM projects WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM keys WHERE purpose = 'test-purpose'");
    }
}
