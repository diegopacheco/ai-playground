package com.github.diegopacheco.adminconsole.audit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;

import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.UUID;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

class AuditServiceTest {
    private AuditRepository repository;
    private AuditService service;
    private final ConnectionConfig connection = new ConnectionConfig(7L, 3L, "prod", ConnectionKind.POSTGRES,
            "localhost", 5432, "shop", null, null, "reader", "secret", null, "diego");

    @BeforeEach
    void setUp() {
        repository = Mockito.mock(AuditRepository.class);
        service = new AuditService(repository);
    }

    private AuditEntry captured() {
        ArgumentCaptor<AuditEntry> captor = ArgumentCaptor.forClass(AuditEntry.class);
        verify(repository).insert(captor.capture());
        return captor.getValue();
    }

    @Test
    void recordsAnAllowedQueryWithItsTimingAndRowCountSoSlowQueriesAreTraceable() {
        UUID queryId = UUID.randomUUID();
        service.allowed(queryId, 1, "diego", connection, "SELECT 1", 42, 10, "127.0.0.1");
        AuditEntry entry = captured();
        assertThat(entry.allowed()).isTrue();
        assertThat(entry.elapsedMs()).isEqualTo(42);
        assertThat(entry.rowCount()).isEqualTo(10);
        assertThat(entry.queryId()).isEqualTo(queryId);
    }

    @Test
    void recordsADeniedStatementBecauseRejectedWritesAreTheMostImportantRowsInTheLog() {
        service.denied(UUID.randomUUID(), "diego", connection, "DROP TABLE users", "write statements are not allowed",
                "127.0.0.1");
        AuditEntry entry = captured();
        assertThat(entry.allowed()).isFalse();
        assertThat(entry.statement()).isEqualTo("DROP TABLE users");
        assertThat(entry.denialReason()).contains("not allowed");
    }

    @Test
    void recordsAFailedQueryWithItsErrorSoBrokenTargetsAreVisible() {
        service.failed(UUID.randomUUID(), 1, "diego", connection, "SELECT 1", 15, "connection refused", "127.0.0.1");
        AuditEntry entry = captured();
        assertThat(entry.error()).isEqualTo("connection refused");
        assertThat(entry.elapsedMs()).isEqualTo(15);
    }

    @Test
    void keepsTheSameQueryIdAcrossPagesSoBrowsingAResultSetReadsAsOneEntry() {
        UUID queryId = UUID.randomUUID();
        service.allowed(queryId, 1, "diego", connection, "SELECT 1", 10, 100, "127.0.0.1");
        service.allowed(queryId, 2, "diego", connection, "SELECT 1", 8, 100, "127.0.0.1");
        ArgumentCaptor<AuditEntry> captor = ArgumentCaptor.forClass(AuditEntry.class);
        verify(repository, Mockito.times(2)).insert(captor.capture());
        assertThat(captor.getAllValues()).extracting(AuditEntry::queryId).containsExactly(queryId, queryId);
        assertThat(captor.getAllValues()).extracting(AuditEntry::page).containsExactly(1, 2);
    }

    @Test
    void neverRecordsConnectionSecretsInTheAuditTrail() {
        service.allowed(UUID.randomUUID(), 1, "diego", connection, "SELECT 1", 5, 1, "127.0.0.1");
        assertThat(captured().toString()).doesNotContain("secret");
    }

    @Test
    void marksADenialAsPageOneSinceARejectedStatementNeverHasFollowOnPages() {
        service.denied(UUID.randomUUID(), "diego", connection, "DELETE FROM users", "denied", "127.0.0.1");
        assertThat(captured().page()).isEqualTo(1);
    }
}
