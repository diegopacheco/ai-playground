package com.github.diegopacheco.devadminconsole.discovery;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ContainerScannerTest {
    @Test
    void readsTheConsolesOwnPortFromItsJdbcUrlSoTheMetadataDatabaseCanBeExcluded() {
        assertThat(ContainerScanner.portOf("jdbc:postgresql://localhost:5433/dev_admin_console")).isEqualTo(5433);
        assertThat(ContainerScanner.portOf("jdbc:postgresql://db.internal:6000/dev_admin_console")).isEqualTo(6000);
    }

    @Test
    void survivesAJdbcUrlWithoutAPortRatherThanThrowingAtStartup() {
        assertThat(ContainerScanner.portOf("jdbc:postgresql://localhost/dev_admin_console")).isZero();
        assertThat(ContainerScanner.portOf(null)).isZero();
        assertThat(ContainerScanner.portOf("nonsense")).isZero();
    }

    @Test
    void readsThePortEvenWhenQueryParametersFollowTheDatabaseName() {
        assertThat(ContainerScanner.portOf("jdbc:postgresql://localhost:5433/dev_admin_console?sslmode=disable"))
                .isEqualTo(5433);
    }
}
