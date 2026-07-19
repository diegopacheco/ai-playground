package com.github.diegopacheco.adminconsole.saved;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import com.github.diegopacheco.adminconsole.project.ConnectionRepository;
import com.github.diegopacheco.adminconsole.project.ProjectRepository;
import javax.sql.DataSource;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

@Tag("integration-test")
@SpringBootTest
class SavedQueryRepositoryIT {
    @Autowired
    private SavedQueryRepository saved;
    @Autowired
    private ProjectRepository projects;
    @Autowired
    private ConnectionRepository connections;
    @Autowired
    private DataSource dataSource;

    private long projectId;
    private long connectionId;

    @BeforeEach
    void setUp() {
        JdbcTemplate jdbc = new JdbcTemplate(dataSource);
        jdbc.update("DELETE FROM saved_queries WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM connections WHERE name LIKE 'it-saved-%'");
        jdbc.update("DELETE FROM projects WHERE name LIKE 'it-saved-%'");
        projectId = projects.create("it-saved-" + System.nanoTime(), "tester").id();
        connectionId = connections.create(new ConnectionConfig(null, projectId, "it-saved-pg", ConnectionKind.POSTGRES,
                "localhost", 5432, "shop", "public", null, "console_reader", "console_reader", null, "tester")).id();
    }

    private SavedQuery query(String name, Long connection) {
        return new SavedQuery(null, projectId, connection, name, "SELECT 1", "postgres", "a note", "tester", null, null);
    }

    @Test
    void savesAQueryForTheWholeProjectSoUsefulQueriesStopLivingInPrivateNotes() {
        SavedQuery created = saved.create(query("it-shared", connectionId));
        assertThat(saved.findByProject(projectId)).extracting(SavedQuery::name).contains("it-shared");
        assertThat(created.createdBy()).isEqualTo("tester");
    }

    @Test
    void pinsAQueryToOneConnectionWhenAsked() {
        SavedQuery created = saved.create(query("it-pinned", connectionId));
        assertThat(saved.findById(created.id()).orElseThrow().connectionId()).isEqualTo(connectionId);
    }

    @Test
    void leavesAQueryUnpinnedSoTheSameSqlCanRunAgainstStagingAndProd() {
        SavedQuery created = saved.create(query("it-loose", null));
        assertThat(saved.findById(created.id()).orElseThrow().connectionId()).isNull();
    }

    @Test
    void anyoneWithProjectAccessCanEditSoASharedLibraryDoesNotGoStale() {
        SavedQuery created = saved.create(query("it-editable", connectionId));
        saved.update(created.id(), "it-editable-renamed", "SELECT 2", "updated note", null);
        SavedQuery reloaded = saved.findById(created.id()).orElseThrow();
        assertThat(reloaded.name()).isEqualTo("it-editable-renamed");
        assertThat(reloaded.statement()).isEqualTo("SELECT 2");
        assertThat(reloaded.connectionId()).isNull();
        assertThat(reloaded.createdBy()).isEqualTo("tester");
    }

    @Test
    void survivesDeletingThePinnedConnectionRatherThanVanishingWithIt() {
        SavedQuery created = saved.create(query("it-orphan", connectionId));
        connections.delete(connectionId);
        SavedQuery reloaded = saved.findById(created.id()).orElseThrow();
        assertThat(reloaded.connectionId()).isNull();
        assertThat(reloaded.statement()).isEqualTo("SELECT 1");
    }

    @Test
    void isRemovedWithItsProject() {
        saved.create(query("it-cascade", connectionId));
        projects.delete(projectId);
        assertThat(saved.findByProject(projectId)).isEmpty();
    }

    @org.junit.jupiter.api.AfterEach
    void cleanUpSoTestDataNeverShowsUpInSomeonesConsole() {
        JdbcTemplate jdbc = new JdbcTemplate(dataSource);
        jdbc.update("DELETE FROM saved_queries WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM connections WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM projects WHERE name LIKE 'it-%'");
        jdbc.update("DELETE FROM keys WHERE purpose = 'test-purpose'");
    }
}
